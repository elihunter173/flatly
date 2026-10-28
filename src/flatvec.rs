//! [`FlatVec`] is a Vec-like data structure that stores elements in a flat array.
//!
//! This means all variable sized data gets written in a flat buffer.

use std::{
    alloc::{self, Layout},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::NonNull,
    slice,
};

use crate::visitor::{
    AlignOfVisitor, FlatError, FlatVisitor, MeasureVisitor, ReadVisitor, WriteVisitor,
};

/// Defines a flat encoding and its value type for each buffer lifetime.
///
/// Implement this trait on a separate schema marker type. For example,
/// `TagSchema` can define `type Value<'buf> = Tag<'buf>`. The same traversal
/// is used to write values and read them back; no schema instance is needed.
/// Zero-byte encodings are checked with debug assertions. They are not tracked
/// as elements because the buffer cursor cannot distinguish them. Zero-sized
/// values with non-empty encoded metadata are supported.
///
/// A local vector cannot produce values requiring a `'static` buffer:
///
/// ```compile_fail
/// use flatly::flatvec::{FlatVec, FlatVecSchema};
/// fn invent_static<S: FlatVecSchema>(values: &FlatVec<S>) -> S::Value<'static> {
///     values.first().unwrap()
/// }
/// ```
///
/// Decoded values cannot escape the vector, whether read directly or iterated:
///
/// ```compile_fail,E0515
/// use flatly::flatvec::{FlatVec, FlatVecSchema};
/// fn escape<'buf, S: FlatVecSchema + 'buf>() -> S::Value<'buf> {
///     let values = FlatVec::<S>::new();
///     values.first().unwrap()
/// }
/// ```
///
/// ```compile_fail,E0515
/// use flatly::flatvec::{FlatVec, FlatVecSchema};
/// fn escape<'buf, S: FlatVecSchema + 'buf>() -> S::Value<'buf> {
///     let values = FlatVec::<S>::new();
///     values.iter().next().unwrap()
/// }
/// ```
///
/// Mutation cannot invalidate a decoded value that is still in use:
///
/// ```compile_fail,E0502
/// use flatly::flatvec::{FlatVec, FlatVecSchema};
/// fn mutate<'buf, S: FlatVecSchema>(
///     values: &'buf mut FlatVec<S>, next: S::Value<'_>,
/// ) -> S::Value<'buf> {
///     let first = values.first().unwrap();
///     values.push(next);
///     first
/// }
/// ```
///
/// # Safety
///
/// `visit` must initialize a valid value on a successful read. Successful traversals must use
/// the same sequence of visitor operations and type parameters, independent of the value,
/// lifetime, and visitor implementation. A traversal may terminate early on a visitor error.
pub unsafe trait FlatVecSchema {
    /// The value type, which may borrow from the encoded buffer for `'buf`.
    ///
    /// No `Self: 'buf` bound is needed: values borrow the buffer, not the schema.
    type Value<'buf>;

    /// Visits every field in the encoded representation without reading from `place` directly.
    ///
    /// Implementations may form raw pointers to fields and pass them to the visitor, but must not
    /// dereference `place` themselves. Alignment discovery passes uninitialized storage and
    /// only inspects the types supplied to visitor operations.
    ///
    /// # Safety
    ///
    /// `place` must be valid for the visitor's operations: writable storage for
    /// a reader or alignment visitor, or an initialized value for a writer or
    /// measurement visitor.
    unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
        place: *mut Self::Value<'buf>,
        v: &mut V,
    ) -> Result<(), FlatError>;
}

/// A Vec-like data structure that stores elements in a flat array.
///
/// `S` is a schema marker implementing [`FlatVecSchema`]. Inputs and decoded
/// elements have type [`S::Value<'buf>`](FlatVecSchema::Value), with decoded
/// references borrowing this vector rather than the original input values.
///
/// Because elements can be variable sized, there is no random access by index.
///
/// # Invariants
///
/// - `count <= len <= capacity`; each tracked element consumes at least one byte.
/// - If `capacity == 0`, `ptr` is dangling. Otherwise, it points to a live allocation.
/// - Bytes `0..len` contain complete encodings of `S`. Field data is initialized,
///   but alignment gaps and padding within scalar or slice elements may not be.
/// - Bytes `len..capacity` are spare storage and may be initialized or uninitialized.
pub struct FlatVec<S: FlatVecSchema> {
    /// Start of the allocation, or [`NonNull::dangling`] when `capacity == 0`.
    ptr: NonNull<u8>,
    /// Number of bytes occupied by encoded elements, including padding.
    len: usize,
    /// Total size of the allocation in bytes.
    capacity: usize,
    /// Number of complete, non-zero-byte element encodings.
    count: usize,
    _marker: PhantomData<S>,
}

impl<S: FlatVecSchema> Default for FlatVec<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: FlatVecSchema> Drop for FlatVec<S> {
    /// Frees the allocation, if there is one.
    fn drop(&mut self) {
        if self.capacity != 0 {
            let layout =
                Layout::from_size_align(self.capacity, Self::alignment()).expect("invalid layout");
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) }
        }
    }
}

impl<S: FlatVecSchema> FlatVec<S> {
    /// Creates an empty `FlatVec` without allocating.
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            count: 0,
            _marker: PhantomData,
        }
    }

    /// Computes the encoded alignment without storing it in the vector.
    fn alignment() -> usize {
        let mut visitor = AlignOfVisitor::new();
        let mut place = MaybeUninit::<S::Value<'_>>::uninit();
        // SAFETY: alignment visitation never reads or writes the supplied storage.
        unsafe { S::visit(place.as_mut_ptr(), &mut visitor) }
            .expect("alignment visitor never returns errors");
        visitor.alignment()
    }

    /// Returns the encoded region (`0..len`), which may contain uninitialized padding.
    fn as_slice(&self) -> &[MaybeUninit<u8>] {
        if self.capacity == 0 {
            &[]
        } else {
            // SAFETY: the encoded region is within the allocation. MaybeUninit
            // permits both initialized field data and uninitialized padding.
            unsafe { slice::from_raw_parts(self.ptr.as_ptr().cast(), self.len) }
        }
    }

    /// Returns spare storage after the encoded region. A failed write may have
    /// initialized part of this storage; callers must not assume it is initialized.
    fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        if self.capacity == 0 {
            &mut []
        } else {
            unsafe {
                let ptr = self.ptr.as_ptr().add(self.len) as *mut MaybeUninit<u8>;
                slice::from_raw_parts_mut(ptr, self.capacity - self.len)
            }
        }
    }

    /// Ensures `required` bytes fit, growing at least geometrically with a
    /// single allocation or reallocation even for a large append.
    ///
    /// # Panics
    ///
    /// Panics if the new layout is invalid or if the allocator returns null.
    fn reserve(&mut self, required: usize) {
        if required <= self.capacity {
            return;
        }
        let new_capacity = self
            .capacity
            .checked_mul(2)
            .unwrap_or(required)
            .max(required)
            .max(8);

        let alignment = Self::alignment();
        let new_layout = Layout::from_size_align(new_capacity, alignment).expect("invalid layout");

        let new_ptr = if self.capacity == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout =
                Layout::from_size_align(self.capacity, alignment).expect("invalid layout");
            unsafe { alloc::realloc(self.ptr.as_ptr(), old_layout, new_capacity) }
        };

        self.ptr = NonNull::new(new_ptr).expect("allocation failed");
        self.capacity = new_capacity;
    }

    /// Measures `v`, reserves sufficient capacity, and writes it once to the back
    /// of `FlatVec`. Capacity grows at least geometrically when necessary.
    ///
    /// Borrowed data is copied into the buffer, so its input lifetime is independent
    /// of the vector and of the schema `S` used to select the encoding.
    ///
    /// # Panics
    ///
    /// Panics if the encoded size overflows, growing the buffer fails, or a slice
    /// in `v` is too long for its encoded length type. Measurement failures leave
    /// the vector unchanged. In debug builds, also panics if the schema encodes
    /// a value without consuming any bytes.
    pub fn push(&mut self, mut v: S::Value<'_>) {
        // Check each value against the alignment discovered from the schema.
        if cfg!(debug_assertions) {
            let mut align_visitor = AlignOfVisitor::new();
            unsafe {
                S::visit(&raw mut v, &mut align_visitor)
                    .expect("alignment visitor never returns errors")
            };
            debug_assert!(
                Self::alignment() == align_visitor.alignment(),
                "FlatVec schema alignment changed between traversals"
            );
        }

        // The allocation base is aligned for every encoded field, so its byte
        // offset gives the same padding as the writer's absolute cursor address.
        let mut measure = MeasureVisitor::from_offset(self.len);
        // SAFETY: `v` is initialized; measurement only reads slice lengths.
        match unsafe { S::visit(&raw mut v, &mut measure) } {
            Ok(()) => {}
            Err(FlatError::InsufficientSpace) => panic!("encoded size overflow"),
            Err(FlatError::LenOverflow) => {
                panic!("slice length doesn't fit in its stored length type");
            }
        }
        let size = measure.size();
        debug_assert!(size != 0, "FlatVec schema encoded no bytes");
        if size == 0 {
            // Zero-byte encodings are not tracked as elements.
            return;
        }
        let new_len = self.len.checked_add(size).expect("encoded size overflow");
        self.reserve(new_len);

        let mut writer = WriteVisitor::from_uninit_bytes(&mut self.spare_capacity_mut()[..size]);
        let start = writer.as_ptr();
        // SAFETY: `v` is initialized and the measured encoding fits in spare
        // storage. The writer copies borrowed data without modifying `v`.
        unsafe { S::visit(&raw mut v, &mut writer) }
            .expect("FlatVec schema write failed after successful measurement");
        let written = writer.as_ptr().addr().get() - start.addr().get();
        assert_eq!(
            written, size,
            "FlatVec schema size changed between measurement and write"
        );
        self.len = new_len;
        // Each counted element consumes at least one byte, so count + 1 is
        // bounded by the validated encoded length.
        self.count += 1;
    }

    /// Returns the first element, or `None` if the vector is empty.
    ///
    /// References in the returned value borrow this vector, preventing it from
    /// being dropped or mutated while those references are still in use.
    pub fn first(&self) -> Option<S::Value<'_>> {
        self.iter().next()
    }

    /// Returns an iterator over the `FlatVec`.
    ///
    /// The iterator yields all elements in insertion order, with references
    /// borrowing this vector for the lifetime of the borrow passed to `iter`.
    /// Its length and size hint report the exact number of remaining elements.
    pub fn iter(&self) -> FlatVecIter<'_, S> {
        FlatVecIter {
            reader: ReadVisitor::from_uninit_bytes(self.as_slice()),
            remaining: self.count,
            _marker: PhantomData,
        }
    }
}

/// Iterator over the elements of a [`FlatVec`].
///
/// Returned by [`FlatVec::iter`].
pub struct FlatVecIter<'buf, S> {
    /// Cursor over the vector's in-use bytes, positioned at the start of the
    /// next element to decode.
    reader: ReadVisitor<'buf>,
    /// Number of encoded elements not yet yielded.
    remaining: usize,
    _marker: PhantomData<S>,
}

impl<'buf, S: FlatVecSchema> Iterator for FlatVecIter<'buf, S> {
    type Item = S::Value<'buf>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let remaining = self.reader.remaining();
        let mut out = MaybeUninit::<S::Value<'_>>::uninit();
        unsafe {
            if S::visit(out.as_mut_ptr(), &mut self.reader).is_err() {
                self.remaining = 0;
                return None;
            }
            debug_assert!(
                self.reader.remaining() < remaining,
                "FlatVec schema decoded no bytes"
            );
            self.remaining -= 1;
            Some(out.assume_init())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<S: FlatVecSchema> ExactSizeIterator for FlatVecIter<'_, S> {}
impl<S: FlatVecSchema> std::iter::FusedIterator for FlatVecIter<'_, S> {}

#[cfg(test)]
mod tests {
    use super::*;

    struct UnitSchema;

    unsafe impl FlatVecSchema for UnitSchema {
        type Value<'buf> = ();

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe { v.scalar(place) }
        }
    }

    struct NoopSchema;

    unsafe impl FlatVecSchema for NoopSchema {
        type Value<'buf> = ();

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            _place: *mut Self::Value<'buf>,
            _v: &mut V,
        ) -> Result<(), FlatError> {
            Ok(())
        }
    }

    struct EmptySliceSchema;

    unsafe impl FlatVecSchema for EmptySliceSchema {
        type Value<'buf> = &'buf [u8];

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe { v.slice::<u8, _>(place) }
        }
    }

    struct UnitWithMetadataSchema;

    unsafe impl FlatVecSchema for UnitWithMetadataSchema {
        type Value<'buf> = ();

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            let mut metadata = 0u8;
            unsafe {
                v.scalar(&raw mut metadata)?;
                v.scalar(place)
            }
        }
    }

    struct UnitSliceSchema;

    unsafe impl FlatVecSchema for UnitSliceSchema {
        type Value<'buf> = &'buf [()];

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe { v.slice::<u8, _>(place) }
        }
    }

    #[test]
    fn test_empty_schemas_do_not_invent_values() {
        let values = FlatVec::<UnitSchema>::new();
        assert_eq!(values.first(), None);
        assert_eq!(values.iter().count(), 0);
        let values = FlatVec::<NoopSchema>::default();
        assert_eq!(values.first(), None);
        assert_eq!(values.iter().count(), 0);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "FlatVec schema encoded no bytes")
    )]
    fn test_zero_sized_scalar_encoding_debug_assert() {
        FlatVec::<UnitSchema>::new().push(());
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "FlatVec schema encoded no bytes")
    )]
    fn test_noop_encoding_debug_assert() {
        FlatVec::<NoopSchema>::new().push(());
    }

    #[test]
    fn test_empty_slice_encodes_length_prefix() {
        let mut values = FlatVec::<EmptySliceSchema>::new();
        values.push(&[]);
        assert_eq!(values.len, 1);
        assert_eq!(values.first(), Some(&[][..]));
        assert_eq!(values.iter().count(), 1);
    }

    #[test]
    fn test_empty_encoding_leaves_vector_empty() {
        let mut values = FlatVec::<UnitSchema>::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| values.push(())));
        assert_eq!(result.is_err(), cfg!(debug_assertions));
        assert_eq!(values.len, 0);
        assert_eq!(values.first(), None);
        assert_eq!(values.iter().len(), 0);
        assert_eq!(values.iter().size_hint(), (0, Some(0)));
        assert_eq!(values.iter().count(), 0);
    }

    #[test]
    fn test_zero_sized_values_with_metadata_are_supported() {
        let mut values = FlatVec::<UnitWithMetadataSchema>::new();
        for _ in 0..3 {
            values.push(());
        }
        assert_eq!(values.first(), Some(()));
        let mut iter = values.iter();
        for _ in 0..3 {
            assert_eq!(iter.next(), Some(()));
        }
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_empty_and_zero_sized_slices_with_length_prefix_are_supported() {
        let mut values = FlatVec::<UnitSliceSchema>::new();
        values.push(&[]);
        values.push(&[(); 3]);
        let lengths: Vec<_> = values.iter().map(|value| value.len()).collect();
        assert_eq!(lengths, [0, 3]);
    }

    /// fixed-sized example
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Point {
        val: f64,
        ts: u64,
    }

    struct PointSchema;

    unsafe impl FlatVecSchema for PointSchema {
        type Value<'buf> = Point;

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe {
                v.scalar(&raw mut (*place).val)?;
                v.scalar(&raw mut (*place).ts)?;
            }
            Ok(())
        }
    }

    #[test]
    fn test_push_and_first() {
        let mut vec = FlatVec::<PointSchema>::new();
        let point = Point { val: 3.14, ts: 255 };
        vec.push(point);
        let retrieved = vec.first().unwrap();
        assert_eq!(retrieved, point);
    }

    #[test]
    fn test_iterator() {
        let mut flatvec = FlatVec::<PointSchema>::new();
        let points = vec![
            Point { val: 1.0, ts: 100 },
            Point { val: 2.0, ts: 200 },
            Point { val: 3.0, ts: 300 },
        ];

        for point in &points {
            flatvec.push(*point);
        }

        let collected: Vec<Point> = flatvec.iter().collect();
        assert_eq!(collected, points);
    }

    #[test]
    fn test_exact_size_iterator_with_variable_sized_elements_and_growth() {
        let inputs = [
            String::new(),
            "short".to_owned(),
            "x".repeat(1024),
            "last".to_owned(),
        ];
        let mut values = FlatVec::<TagSchema>::new();
        assert_eq!(values.iter().len(), 0);
        assert_eq!(values.iter().size_hint(), (0, Some(0)));
        for (i, input) in inputs.iter().enumerate() {
            values.push(Tag {
                val: input,
                ts: i as f64,
            });
            assert_eq!(values.iter().len(), i + 1);
        }

        let mut iter = values.iter();
        for (i, input) in inputs.iter().enumerate() {
            let remaining = inputs.len() - i;
            assert_eq!(iter.len(), remaining);
            assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
            assert_eq!(
                iter.next(),
                Some(Tag {
                    val: input,
                    ts: i as f64
                })
            );
        }
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(values.iter().len(), inputs.len());

        let mut skipped = values.iter();
        assert_eq!(skipped.nth(1).unwrap().val, inputs[1]);
        assert_eq!(skipped.len(), 2);
        assert_eq!(skipped.size_hint(), (2, Some(2)));
        assert_eq!(skipped.nth(10), None);
        assert_eq!(skipped.len(), 0);
        assert_eq!(skipped.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_measured_appends_write_once_and_preserve_unaligned_row_offsets() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static WRITES: AtomicUsize = AtomicUsize::new(0);
        struct CountedSchema;

        unsafe impl FlatVecSchema for CountedSchema {
            type Value<'buf> = (&'buf [u8], u64, u8);

            unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
                place: *mut Self::Value<'buf>,
                v: &mut V,
            ) -> Result<(), FlatError> {
                // Instrument traversal without changing the encoding for any visitor.
                if std::any::type_name::<V>() == std::any::type_name::<WriteVisitor<'_>>() {
                    WRITES.fetch_add(1, Ordering::Relaxed);
                }
                unsafe {
                    v.slice::<u16, _>(&raw mut (*place).0)?;
                    v.scalar(&raw mut (*place).1)?;
                    v.scalar(&raw mut (*place).2)
                }
            }
        }

        WRITES.store(0, Ordering::Relaxed);
        let mut values = FlatVec::<CountedSchema>::new();
        values.push((b"abc", 1, 10));
        assert_eq!(WRITES.load(Ordering::Relaxed), 1);
        assert_eq!(values.len % align_of::<u64>(), 1);

        let large = vec![42; 32_768];
        values.push((&large, 2, 20));
        assert_eq!(WRITES.load(Ordering::Relaxed), 2);
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [(&b"abc"[..], 1, 10), (&large[..], 2, 20),]
        );

        // An invalid length is rejected before reserving or writing anything,
        // including when earlier elements have already been committed.
        let previous = (values.ptr, values.len, values.capacity);
        let too_long = vec![0; 65_536];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values.push((&too_long, 999, 99));
        }));
        assert!(result.is_err());
        assert_eq!((values.ptr, values.len, values.capacity), previous);
        assert_eq!(values.iter().len(), 2);
        assert_eq!(values.iter().size_hint(), (2, Some(2)));
        assert_eq!(WRITES.load(Ordering::Relaxed), 2);

        values.push((b"last", 3, 30));
        assert_eq!(WRITES.load(Ordering::Relaxed), 3);
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [
                (&b"abc"[..], 1, 10),
                (&large[..], 2, 20),
                (&b"last"[..], 3, 30),
            ]
        );
    }

    #[test]
    fn test_invalid_first_append_does_not_allocate() {
        let mut values = FlatVec::<UnitSliceSchema>::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values.push(&[(); 256]);
        }));
        assert!(result.is_err());
        assert_eq!(values.capacity, 0);
        assert_eq!(values.len, 0);
        assert_eq!(values.first(), None);
        values.push(&[(); 3]);
        assert_eq!(values.first().unwrap().len(), 3);
    }

    #[test]
    fn test_empty_iterator() {
        let vec = FlatVec::<PointSchema>::new();
        let collected: Vec<Point> = vec.iter().collect();
        assert_eq!(collected.len(), 0);
    }

    #[repr(align(64))]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Overaligned(u64);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MixedAlignment {
        prefix: u8,
        value: Overaligned,
        suffix: u32,
    }

    struct MixedAlignmentSchema;

    unsafe impl FlatVecSchema for MixedAlignmentSchema {
        type Value<'buf> = MixedAlignment;

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe {
                v.scalar(&raw mut (*place).prefix)?;
                v.scalar(&raw mut (*place).value)?;
                v.scalar(&raw mut (*place).suffix)?;
            }
            Ok(())
        }
    }

    struct OveralignedSliceSchema;

    unsafe impl FlatVecSchema for OveralignedSliceSchema {
        type Value<'buf> = &'buf [Overaligned];

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe { v.slice::<u8, _>(place) }
        }
    }

    #[test]
    fn test_overaligned_slices_including_empty_survive_growth() {
        let mut values = FlatVec::<OveralignedSliceSchema>::new();
        assert_eq!(FlatVec::<OveralignedSliceSchema>::alignment(), 64);
        values.push(&[]);
        let initial_capacity = values.capacity;
        for i in 0..20 {
            values.push(&[Overaligned(i)]);
        }
        assert!(values.capacity > initial_capacity);
        assert_eq!(values.ptr.as_ptr().addr() % 64, 0);

        let mut iter = values.iter();
        let empty = iter.next().unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.as_ptr().addr() % 64, 0);
        for i in 0..20 {
            let slice = iter.next().unwrap();
            assert_eq!(slice, &[Overaligned(i)]);
            assert_eq!(slice.as_ptr().addr() % 64, 0);
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "FlatVec schema alignment changed between traversals")]
    fn test_changed_schema_alignment_debug_assert() {
        use std::sync::atomic::{AtomicBool, Ordering};

        static INCLUDE_ALIGNED: AtomicBool = AtomicBool::new(false);
        struct ChangingSchema;

        // Deliberately violates the traversal contract to exercise the debug check.
        unsafe impl FlatVecSchema for ChangingSchema {
            type Value<'buf> = (u8, Overaligned);

            unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
                place: *mut Self::Value<'buf>,
                v: &mut V,
            ) -> Result<(), FlatError> {
                unsafe {
                    v.scalar(&raw mut (*place).0)?;
                    if INCLUDE_ALIGNED.swap(true, Ordering::Relaxed) {
                        v.scalar(&raw mut (*place).1)?;
                    }
                }
                Ok(())
            }
        }

        INCLUDE_ALIGNED.store(false, Ordering::Relaxed);
        let mut values = FlatVec::<ChangingSchema>::new();
        values.push((0, Overaligned(0)));
    }

    #[test]
    fn test_overaligned_values_survive_growth() {
        let values: Vec<_> = (0..20)
            .map(|i| MixedAlignment {
                prefix: i as u8,
                value: Overaligned(i),
                suffix: i as u32 * 10,
            })
            .collect();
        let mut flatvec = FlatVec::<MixedAlignmentSchema>::new();
        for value in values.iter().copied() {
            flatvec.push(value);
        }

        let collected: Vec<_> = flatvec.iter().collect();
        assert_eq!(collected, values);
    }

    /// variable-sized example
    #[derive(Debug, PartialEq, Clone)]
    struct Tag<'buf> {
        val: &'buf str,
        ts: f64,
    }

    struct TagSchema;

    unsafe impl FlatVecSchema for TagSchema {
        type Value<'buf> = Tag<'buf>;

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe {
                // `&str` has the same layout as `&[u8]`.
                let val = (&raw mut (*place).val).cast::<&'buf [u8]>();
                v.slice::<u32, _>(val)?;
                v.scalar(&raw mut (*place).ts)?;
            }
            Ok(())
        }
    }

    #[test]
    fn test_tag_copies_borrowed_input() {
        let input = String::from("temporary input");
        let long_input = "x".repeat(1024);
        let mut flatvec = FlatVec::<TagSchema>::new();
        flatvec.push(Tag {
            val: &input,
            ts: 1.0,
        });
        flatvec.push(Tag {
            val: &long_input,
            ts: 2.0,
        });

        let first = flatvec.first().unwrap();
        assert_eq!(first.val, input);
        assert_ne!(first.val.as_ptr(), input.as_ptr());
        let mut iter = flatvec.iter();
        let from_iter = iter.next().unwrap();
        drop(iter);
        assert_eq!(from_iter, first);

        let collected: Vec<_> = flatvec.iter().collect();
        assert_eq!(collected[0].val, input);
        assert_eq!(collected[1].val, long_input);
    }

    #[test]
    fn test_tag_read_push_read_with_short_lived_inputs() {
        let mut flatvec = FlatVec::<TagSchema>::new();
        {
            let input = String::from("hello");
            flatvec.push(Tag {
                val: &input,
                ts: 1.0,
            });
        }
        assert_eq!(flatvec.first().unwrap().val, "hello");
        assert_eq!(flatvec.iter().next().unwrap().val, "hello");

        // Both the input borrow and the previous buffer borrows have ended.
        // Force growth, then borrow the vector again with a fresh value lifetime.
        let capacity = flatvec.capacity;
        {
            let input = "x".repeat(capacity + 1);
            flatvec.push(Tag {
                val: &input,
                ts: 2.0,
            });
        }
        assert!(flatvec.capacity > capacity);
        assert_eq!(flatvec.first().unwrap().val, "hello");
        let collected: Vec<_> = flatvec.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[1].val.len(), capacity + 1);
    }

    #[test]
    fn test_owned_value_can_outlive_vector() {
        let point = {
            let mut flatvec = FlatVec::<PointSchema>::new();
            flatvec.push(Point { val: 1.0, ts: 2 });
            flatvec.first().unwrap()
        };
        assert_eq!(point, Point { val: 1.0, ts: 2 });
    }

    #[test]
    fn test_tag_with_string() {
        let mut flatvec = FlatVec::<TagSchema>::new();
        let tags = vec![
            Tag {
                val: "hello",
                ts: 10.0,
            },
            Tag {
                val: "world",
                ts: 20.0,
            },
            Tag {
                val: "rust!",
                ts: 30.0,
            },
        ];

        for tag in &tags {
            flatvec.push(tag.clone());
        }

        let collected: Vec<_> = flatvec.iter().collect();
        assert_eq!(collected, tags);
    }

    /// Slices of non-byte elements, plus a nested value.
    #[derive(Debug, PartialEq, Clone)]
    struct Series<'buf> {
        name: Tag<'buf>,
        points: &'buf [Point],
        flag: u8,
    }

    struct SeriesSchema;

    unsafe impl FlatVecSchema for SeriesSchema {
        type Value<'buf> = Series<'buf>;

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe {
                TagSchema::visit(&raw mut (*place).name, v)?;

                let points = &raw mut (*place).points;
                v.slice::<u16, _>(points)?;

                v.scalar(&raw mut (*place).flag)?;
            }
            Ok(())
        }
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Padded {
        prefix: u8,
        value: u64,
    }

    struct PaddedSchema;

    unsafe impl FlatVecSchema for PaddedSchema {
        type Value<'buf> = (Padded, &'buf [Padded], u8);

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe {
                // Copy a padded scalar, then a slice of padded elements. The
                // length prefix and trailing byte also create alignment gaps.
                v.scalar(&raw mut (*place).0)?;
                let items = &raw mut (*place).1;
                v.slice::<u16, _>(items)?;
                v.scalar(&raw mut (*place).2)
            }
        }
    }

    #[test]
    fn test_uninitialized_padding_survives_growth_and_typed_reads() {
        assert!(size_of::<Padded>() > size_of::<u8>() + size_of::<u64>());
        let mut flatvec = FlatVec::<PaddedSchema>::new();
        for i in 0..20 {
            let scalar = Padded {
                prefix: i,
                value: u64::from(i) * 10,
            };
            let items = [
                scalar,
                Padded {
                    prefix: i + 1,
                    value: u64::from(i) * 10 + 1,
                },
            ];
            flatvec.push((scalar, &items, i));
        }

        // This type assertion guards the buffer boundary: exposing &[u8] here
        // would wrongly promise initialized alignment and struct padding.
        let encoded: &[MaybeUninit<u8>] = flatvec.as_slice();
        assert_eq!(encoded.len(), flatvec.len);

        let (scalar, items, suffix) = flatvec.first().unwrap();
        assert_eq!(
            scalar,
            Padded {
                prefix: 0,
                value: 0
            }
        );
        assert_eq!(items.len(), 2);
        assert_eq!(suffix, 0);
        let mut count = 0;
        for (i, (scalar, items, suffix)) in flatvec.iter().enumerate() {
            assert_eq!(
                scalar,
                Padded {
                    prefix: i as u8,
                    value: i as u64 * 10
                }
            );
            assert_eq!(
                items,
                &[
                    scalar,
                    Padded {
                        prefix: i as u8 + 1,
                        value: i as u64 * 10 + 1
                    }
                ]
            );
            assert_eq!(suffix, i as u8);
            count += 1;
        }
        assert_eq!(count, 20);
    }

    #[test]
    fn test_series_with_slices() {
        let pts = [Point { val: 1.0, ts: 1 }, Point { val: 2.0, ts: 2 }];
        let series = vec![
            Series {
                name: Tag { val: "a", ts: 0.5 },
                points: &pts,
                flag: 1,
            },
            Series {
                name: Tag { val: "", ts: 1.5 },
                points: &[],
                flag: 2,
            },
            Series {
                name: Tag {
                    val: "a much longer name to force growth",
                    ts: 2.5,
                },
                points: &pts[1..],
                flag: 3,
            },
        ];

        let mut flatvec = FlatVec::<SeriesSchema>::new();
        for s in &series {
            flatvec.push(s.clone());
        }

        let collected: Vec<_> = flatvec.iter().collect();
        assert_eq!(collected, series);
    }
}
