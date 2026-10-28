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

use crate::visitor::{AlignOfVisitor, FlatError, FlatVisitor, ReadVisitor, WriteVisitor};

/// Defines a flat encoding and its value type for each buffer lifetime.
///
/// Implement this trait on a separate schema marker type. For example,
/// `TagSchema` can define `type Value<'buf> = Tag<'buf>`. The same traversal
/// is used to write values and read them back; no schema instance is needed.
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
/// `ALIGN` must be a valid allocation alignment and the maximum alignment used
/// by the encoding. `visit` must visit every field, initializing a valid value on
/// a successful read, and use the same encoding for every lifetime. Borrowed data
/// must be visited through lifetime-bound slice operations, not copied as
/// reference-valued scalars. Scalar and slice element types must be valid to copy
/// into the buffer and back without retaining references to the original value.
pub unsafe trait FlatVecSchema {
    const ALIGN: usize;

    /// The value type, which may borrow from the encoded buffer for `'buf`.
    ///
    /// No `Self: 'buf` bound is needed: values borrow the buffer, not the schema.
    type Value<'buf>;

    /// Visits every field in the encoded representation without reading from `place` directly.
    ///
    /// Implementations may form raw pointers to fields and pass them to the visitor, but must not
    /// dereference `place` themselves.
    ///
    /// # Safety
    ///
    /// `place` must be valid for the visitor's operations: writable storage for
    /// a reader, or an initialized value for a writer.
    unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
        place: *mut Self::Value<'buf>,
        v: &mut V,
    ) -> Result<(), FlatError>;
}

pub(crate) fn read_from<'buf, S: FlatVecSchema>(
    visitor: &mut ReadVisitor<'buf>,
) -> Result<S::Value<'buf>, FlatError> {
    let mut out = MaybeUninit::<S::Value<'buf>>::uninit();
    // SAFETY: `out` is valid for writes; the `FlatVecSchema` contract guarantees every field is
    // initialized on success. On error `out` is dropped as `MaybeUninit`, so no
    // uninitialized fields are dropped.
    unsafe {
        S::visit(out.as_mut_ptr(), visitor)?;
        Ok(out.assume_init())
    }
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
/// - `len <= capacity`.
/// - If `capacity == 0`, `ptr` is dangling. Otherwise, it points to a live allocation.
/// - Bytes `0..len` are in use and `len..capacity` are uninitialized.
pub struct FlatVec<S: FlatVecSchema> {
    /// Start of the allocation, or [`NonNull::dangling`] when `capacity == 0`.
    ptr: NonNull<u8>,
    /// Number of bytes at the start of the buffer are initialized.
    len: usize,
    /// Total size of the allocation in bytes.
    capacity: usize,
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
            let layout = Layout::from_size_align(self.capacity, S::ALIGN).expect("invalid layout");
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
            _marker: PhantomData,
        }
    }

    /// Returns the initialized bytes (`0..len`) as a slice.
    fn as_slice(&self) -> &[u8] {
        if self.capacity == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Returns the tail of uninitialized bytes at the end of Buf's allocation.
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

    /// Grows the allocation to 8 bytes if nothing has been allocated yet,
    /// otherwise to twice the current capacity.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize` or if the allocator returns null.
    fn double(&mut self) {
        let new_capacity = if self.capacity == 0 {
            8
        } else {
            self.capacity.checked_mul(2).expect("capacity overflow")
        };

        let new_layout = Layout::from_size_align(new_capacity, S::ALIGN).expect("invalid layout");

        let new_ptr = if self.capacity == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            // TODO: Realloc needs to preserve the alignment of the current buffer.
            let old_layout =
                Layout::from_size_align(self.capacity, S::ALIGN).expect("invalid layout");
            unsafe { alloc::realloc(self.ptr.as_ptr(), old_layout, new_capacity) }
        };

        self.ptr = NonNull::new(new_ptr).expect("allocation failed");
        self.capacity = new_capacity;
    }

    /// Sets the number of in-use bytes to `new_len`.
    ///
    /// # Safety
    ///
    /// `new_len` must be `<= capacity`, and bytes `0..new_len` must be initialized.
    unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    /// Appends `v` to the back of `FlatVec`. If item doesn't fit, the buffer size is doubled.
    ///
    /// Borrowed data is copied into the buffer, so its input lifetime is independent
    /// of the vector and of the schema `S` used to select the encoding.
    ///
    /// # Panics
    ///
    /// Panics if growing the buffer fails, or if a slice in `v` is too long for the length type
    /// its [`FlatVecSchema`] impl stores it with.
    pub fn push(&mut self, mut v: S::Value<'_>) {
        // Ensure S::ALIGN is right
        if cfg!(debug_assertions) {
            let mut align_visitor = AlignOfVisitor::new();
            unsafe {
                S::visit(&raw mut v, &mut align_visitor)
                    .expect("align of visitor never returns errors")
            };
            assert_eq!(S::ALIGN, align_visitor.alignment());
        }

        loop {
            let mut writer = WriteVisitor::from_uninit_bytes(self.spare_capacity_mut());
            // SAFETY: `v` is a live, initialized value; the write visitor only reads through it.
            match unsafe { S::visit(&raw mut v, &mut writer) } {
                Ok(()) => {
                    let write_addr = writer.as_ptr().addr().get();
                    let new_len = write_addr - self.ptr.as_ptr().addr();
                    unsafe { self.set_len(new_len) };
                    break;
                }
                Err(FlatError::InsufficientSpace) => {
                    self.double();
                }
                Err(FlatError::LenOverflow) => {
                    panic!("slice length doesn't fit in its stored length type");
                }
            }
        }
    }

    /// Returns the first element, or `None` if the vector is empty.
    ///
    /// References in the returned value borrow this vector, preventing it from
    /// being dropped or mutated while those references are still in use.
    pub fn first(&self) -> Option<S::Value<'_>> {
        let mut visitor = ReadVisitor::from_bytes(self.as_slice());
        read_from::<S>(&mut visitor).ok()
    }

    /// Returns an iterator over the `FlatVec`.
    ///
    /// The iterator yields all elements in insertion order, with references
    /// borrowing this vector for the lifetime of the borrow passed to `iter`.
    pub fn iter(&self) -> FlatVecIter<'_, S> {
        FlatVecIter {
            reader: ReadVisitor::from_bytes(self.as_slice()),
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
    _marker: PhantomData<S>,
}

impl<'buf, S: FlatVecSchema> Iterator for FlatVecIter<'buf, S> {
    type Item = S::Value<'buf>;

    fn next(&mut self) -> Option<Self::Item> {
        read_from::<S>(&mut self.reader).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fixed-sized example
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Point {
        val: f64,
        ts: u64,
    }

    struct PointSchema;

    unsafe impl FlatVecSchema for PointSchema {
        const ALIGN: usize = align_of::<Point>();
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
        const ALIGN: usize = align_of::<Overaligned>();
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
        const ALIGN: usize = align_of::<f64>();
        type Value<'buf> = Tag<'buf>;

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe {
                // `&str` has the same layout as `&[u8]`.
                let val = (&raw mut (*place).val).cast::<&'buf [u8]>();
                let len = v.slice_len::<u32, _>(val)?;
                v.slice_data(val, len)?;
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
        const ALIGN: usize = align_of::<Point>();
        type Value<'buf> = Series<'buf>;

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut V,
        ) -> Result<(), FlatError> {
            unsafe {
                TagSchema::visit(&raw mut (*place).name, v)?;

                let points = &raw mut (*place).points;
                let len = v.slice_len::<u16, _>(points)?;
                v.slice_data(points, len)?;

                v.scalar(&raw mut (*place).flag)?;
            }
            Ok(())
        }
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
