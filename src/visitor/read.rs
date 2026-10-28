use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull, slice};

use super::{FlatError, FlatLen, FlatVisitor};

/// Reads values and borrowed slices from an encoded buffer.
///
/// The buffer may contain uninitialized alignment gaps and padding within
/// encoded types. Unsafe read operations must only access valid, initialized
/// values of the requested types; padding need not be initialized.
///
/// A destination slice cannot outlive that buffer, even through a raw pointer:
///
/// ```compile_fail
/// use std::mem::MaybeUninit;
/// use flatly::visitor::{FlatVisitor, ReadVisitor};
/// unsafe fn invent_static(reader: &mut ReadVisitor<'_>) -> &'static [u8] {
///     let mut out = MaybeUninit::<&'static [u8]>::uninit();
///     unsafe {
///         reader.slice::<u32, _>(out.as_mut_ptr()).unwrap();
///         out.assume_init()
///     }
/// }
/// ```
pub struct ReadVisitor<'buf> {
    start: NonNull<u8>,
    end: NonNull<u8>,
    _lifetime: PhantomData<&'buf [MaybeUninit<u8>]>,
}

impl<'buf> ReadVisitor<'buf> {
    pub(crate) fn from_uninit_bytes(bytes: &'buf [MaybeUninit<u8>]) -> Self {
        let range = bytes.as_ptr_range();
        Self {
            start: NonNull::new(range.start.cast_mut().cast()).unwrap(),
            end: NonNull::new(range.end.cast_mut().cast()).unwrap(),
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.end.addr().get() - self.start.addr().get()
    }

    fn take_array<T>(&mut self, len: usize) -> Result<NonNull<T>, FlatError> {
        let size = size_of::<T>()
            .checked_mul(len)
            .ok_or(FlatError::InsufficientSpace)?;
        let pad = self.start.align_offset(align_of::<T>());
        let remaining = self.remaining();
        let needed = pad.checked_add(size).ok_or(FlatError::InsufficientSpace)?;
        if needed > remaining {
            return Err(FlatError::InsufficientSpace);
        }
        unsafe {
            let ptr = self.start.add(pad);
            self.start = ptr.add(size);
            Ok(ptr.cast())
        }
    }
}

impl<'buf> FlatVisitor<'buf> for ReadVisitor<'buf> {
    unsafe fn scalar<T: Copy + 'static>(&mut self, place: *mut T) -> Result<(), FlatError> {
        unsafe { place.write(self.take_array::<T>(1)?.read()) };
        Ok(())
    }

    unsafe fn slice<L: FlatLen, T: Copy + 'static>(
        &mut self,
        place: *mut &'buf [T],
    ) -> Result<(), FlatError> {
        let mut stored = MaybeUninit::<L>::uninit();
        unsafe {
            self.scalar(stored.as_mut_ptr())?;
            let len = stored
                .assume_init()
                .to_usize()
                .ok_or(FlatError::LenOverflow)?;
            let head = self.take_array::<T>(len)?;
            place.write(slice::from_raw_parts(head.as_ptr(), len));
        }
        Ok(())
    }
}
