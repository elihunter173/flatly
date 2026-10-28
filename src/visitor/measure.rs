use super::{FlatError, FlatLen, FlatVisitor};

/// Counts encoded bytes without writing or copying payload data.
///
/// Includes scalar padding, slice length prefixes, and slice element padding,
/// matching [`super::WriteVisitor`]. Slice operations read the input slice's
/// length, so slice references must be initialized; elements are not inspected.
/// No input borrows are retained. Returns [`FlatError::LenOverflow`] if a slice
/// length cannot fit its prefix type, or [`FlatError::InsufficientSpace`] if size
/// arithmetic overflows. After an error, the count may include a partial encoding.
///
/// Offsets are relative to a buffer base aligned to every type visited (as
/// determined by [`super::AlignOfVisitor`]). For another base address, pass its
/// absolute starting address to [`Self::from_offset`] instead.
#[derive(Default)]
pub struct MeasureVisitor {
    start: usize,
    offset: usize,
}

impl MeasureVisitor {
    /// Starts measuring at an existing buffer offset, including any leading
    /// padding needed there. [`Self::size`] reports only the additional bytes.
    pub fn from_offset(offset: usize) -> Self {
        Self {
            start: offset,
            offset,
        }
    }

    /// Returns the number of bytes consumed since the starting offset.
    pub fn size(&self) -> usize {
        self.offset - self.start
    }

    fn measure_array<T>(&mut self, len: usize) -> Result<(), FlatError> {
        let size = size_of::<T>()
            .checked_mul(len)
            .ok_or(FlatError::InsufficientSpace)?;
        let alignment = align_of::<T>();
        let padding = (alignment - self.offset % alignment) % alignment;
        self.offset = self
            .offset
            .checked_add(padding)
            .and_then(|offset| offset.checked_add(size))
            .ok_or(FlatError::InsufficientSpace)?;
        Ok(())
    }
}

impl<'buf> FlatVisitor<'buf> for MeasureVisitor {
    unsafe fn scalar<T: Copy + 'static>(&mut self, _place: *mut T) -> Result<(), FlatError> {
        self.measure_array::<T>(1)
    }

    unsafe fn slice<L: FlatLen, T: Copy + 'static>(
        &mut self,
        place: *mut &'buf [T],
    ) -> Result<(), FlatError> {
        // SAFETY: the caller supplies an initialized slice reference. Only its
        // length is read; neither the reference nor its elements are modified.
        let len = unsafe { place.read().len() };
        L::from_usize(len).ok_or(FlatError::LenOverflow)?;
        self.measure_array::<L>(1)?;
        self.measure_array::<T>(len)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::*;
    use crate::visitor::WriteVisitor;

    #[repr(align(64))]
    struct Buffer([MaybeUninit<u8>; 2048]);

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Padded {
        prefix: u8,
        value: u64,
    }

    #[repr(align(64))]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Aligned(u8);

    #[repr(align(64))]
    #[derive(Clone, Copy)]
    struct AlignedUnit;

    struct Row<'buf> {
        prefix: u8,
        scalar: Padded,
        bytes: &'buf [u8],
        padded: &'buf [Padded],
        aligned: &'buf [Aligned],
        units: &'buf [AlignedUnit],
        suffix: u16,
    }

    unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
        place: *mut Row<'buf>,
        visitor: &mut V,
    ) -> Result<(), FlatError> {
        unsafe {
            visitor.scalar(&raw mut (*place).prefix)?;
            visitor.scalar(&raw mut (*place).scalar)?;
            visitor.slice::<u16, _>(&raw mut (*place).bytes)?;
            visitor.slice::<u8, _>(&raw mut (*place).padded)?;
            visitor.slice::<u32, _>(&raw mut (*place).aligned)?;
            visitor.slice::<u8, _>(&raw mut (*place).units)?;
            visitor.scalar(&raw mut (*place).suffix)
        }
    }

    #[test]
    fn matches_writer_for_aligned_and_unaligned_offsets_and_empty_slices() {
        for offset in 0..128 {
            for empty in [false, true] {
                let padded = [Padded {
                    prefix: 2,
                    value: 42,
                }; 2];
                let aligned = [Aligned(3); 3];
                let mut row = Row {
                    prefix: 1,
                    scalar: padded[0],
                    bytes: if empty { &[] } else { b"hello" },
                    padded: if empty { &[] } else { &padded },
                    aligned: if empty { &[] } else { &aligned },
                    units: if empty { &[] } else { &[AlignedUnit; 5] },
                    suffix: 99,
                };
                let mut measure = MeasureVisitor::from_offset(offset);
                unsafe { visit(&raw mut row, &mut measure).unwrap() };

                let mut buffer = Buffer([MaybeUninit::uninit(); 2048]);
                let start = buffer.0.as_ptr().addr() + offset;
                let mut writer = WriteVisitor::from_uninit_bytes(&mut buffer.0[offset..]);
                unsafe { visit(&raw mut row, &mut writer).unwrap() };
                assert_eq!(measure.size(), writer.as_ptr().addr().get() - start);
                // Measuring and writing leave the input untouched.
                assert_eq!(row.scalar, padded[0]);
                assert_eq!(row.bytes, if empty { &b""[..] } else { &b"hello"[..] });
                assert_eq!(row.suffix, 99);
            }
        }
    }

    #[test]
    fn counts_multiple_values_and_alignment_padding() {
        let mut measure = MeasureVisitor::default();
        assert_eq!(measure.size(), 0);
        let mut byte = 1u8;
        let mut scalar = 2u64;
        unsafe {
            measure.scalar(&raw mut byte).unwrap();
            assert_eq!(measure.size(), 1);
            measure.scalar(&raw mut scalar).unwrap();
            assert_eq!(measure.size(), 16);
            measure.scalar(&raw mut byte).unwrap();
            assert_eq!(measure.size(), 17);
            measure.scalar(&raw mut scalar).unwrap();
        }
        assert_eq!(measure.size(), 32);
        assert_eq!(MeasureVisitor::default().size(), 0);
    }

    #[test]
    fn empty_and_zero_sized_payloads_still_include_prefix_and_padding() {
        let mut empty: &[Aligned] = &[];
        let mut measure = MeasureVisitor::default();
        unsafe { measure.slice::<u8, _>(&raw mut empty).unwrap() };
        assert_eq!(measure.size(), 64);

        let mut units: &[AlignedUnit] = &[AlignedUnit; 3];
        let mut measure = MeasureVisitor::from_offset(1);
        unsafe { measure.slice::<u8, _>(&raw mut units).unwrap() };
        assert_eq!(measure.size(), 63);

        let mut unit = AlignedUnit;
        let mut measure = MeasureVisitor::from_offset(1);
        unsafe { measure.scalar(&raw mut unit).unwrap() };
        assert_eq!(measure.size(), 63);
    }

    #[test]
    fn scalar_measurement_does_not_read_the_value() {
        let mut value = MaybeUninit::<Padded>::uninit();
        let mut measure = MeasureVisitor::default();
        unsafe { measure.scalar(value.as_mut_ptr()).unwrap() };
        assert_eq!(measure.size(), size_of::<Padded>());
    }

    #[test]
    fn rejects_unrepresentable_slice_lengths_before_advancing() {
        let bytes = [0u8; 300];
        let mut input: &[u8] = &bytes;
        let mut measure = MeasureVisitor::from_offset(1);
        assert_eq!(
            unsafe { measure.slice::<u8, _>(&raw mut input) },
            Err(FlatError::LenOverflow)
        );
        assert_eq!(measure.size(), 0);
        unsafe { measure.slice::<u16, _>(&raw mut input).unwrap() };
        assert_eq!(measure.size(), 303); // One padding byte, u16 length, and payload.
    }

    #[test]
    fn rejects_arithmetic_overflow_without_wrapping() {
        let mut value = 0u8;
        let mut measure = MeasureVisitor::from_offset(usize::MAX);
        assert_eq!(
            unsafe { measure.scalar(&raw mut value) },
            Err(FlatError::InsufficientSpace)
        );
        assert_eq!(measure.size(), 0);

        let mut aligned = AlignedUnit;
        assert_eq!(
            unsafe { measure.scalar(&raw mut aligned) },
            Err(FlatError::InsufficientSpace)
        );
        assert_eq!(measure.size(), 0);

        let mut measure = MeasureVisitor::default();
        assert_eq!(
            measure.measure_array::<u64>(usize::MAX),
            Err(FlatError::InsufficientSpace)
        );
        assert_eq!(measure.size(), 0);
    }
}
