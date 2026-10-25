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

use crate::{FlatRead, FlatReader, FlatWrite, FlatWriter, reader_writer::InsufficientSpace};

/// A manually managed, growable byte buffer with no alignment guarantees.
///
///
/// # Invariants
///
/// - `len <= capacity`.
/// - If `capacity == 0`, `ptr` is dangling. Otherwise, it points to a live allocation.
/// - Bytes `0..len` are in use and `len..capacity` are uninitialized.
struct Buf {
    /// Start of the allocation, or [`NonNull::dangling`] when `capacity == 0`.
    ptr: NonNull<u8>,
    /// Number of bytes at the start of the buffer are initialized.
    len: usize,
    /// Total size of the allocation in bytes.
    capacity: usize,
}

impl Buf {
    /// Creates an empty buffer without allocating.
    fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
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

        let new_layout = Layout::array::<u8>(new_capacity).expect("invalid layout");

        let new_ptr = if self.capacity == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            // TODO: Realloc needs to preserve the alignment of the current buffer.
            let old_layout = Layout::array::<u8>(self.capacity).expect("invalid layout");
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

    /// Returns the raw base pointer of the buffer. This pointer is dangling
    /// when nothing has been allocated.
    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Buf {
    /// Frees the allocation, if there is one.
    fn drop(&mut self) {
        if self.capacity != 0 {
            let layout = Layout::array::<u8>(self.capacity).expect("invalid layout");
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) }
        }
    }
}

/// A Vec-like data structure that stores stores elements in a flat array.
///
/// Elements are inserted via their [`FlatWrite`] implementation and read via their [`FlatRead`]
/// implmentation.
///
/// Because elements can be variable sized, there is no random access by index.
pub struct FlatVec<T> {
    /// Storage for all elements stored back to back.
    data: Buf,
    _marker: PhantomData<T>,
}

impl<T> FlatVec<T> {
    /// Creates an empty `FlatVec` without allocating.
    pub fn new() -> Self {
        Self {
            data: Buf::new(),
            _marker: PhantomData,
        }
    }
}

impl<T> Default for FlatVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FlatWrite> FlatVec<T> {
    /// Appends `v` to the back of `FlatVec`. If item doesn't fit, the buffer size is doubled.
    ///
    /// # Panics
    ///
    /// Panics if growing the buffer fails.
    pub fn push(&mut self, v: T) {
        loop {
            let mut writer = FlatWriter::from_uninit_bytes(self.data.spare_capacity_mut());
            match v.write_into(&mut writer) {
                Ok(()) => {
                    let write_addr = writer.as_ptr().addr().get();
                    let new_len = write_addr - self.data.as_ptr().addr();
                    unsafe { self.data.set_len(new_len) };
                    break;
                }
                Err(InsufficientSpace) => {
                    self.data.double();
                }
            }
        }
    }
}

impl<'buf, T: FlatRead<'buf>> FlatVec<T> {
    /// Returns the first element, or `None` if the vector is empty.
    pub fn first(&'buf self) -> Option<T> {
        let mut reader = FlatReader::from_bytes(self.data.as_slice());
        T::read_from(&mut reader).ok()
    }

    /// Returns an iterator over the `FlatVec`.
    ///
    /// The iterator yields all elements in insertion order.
    pub fn iter(&'buf self) -> FlatVecIter<'buf, T> {
        FlatVecIter {
            reader: FlatReader::from_bytes(self.data.as_slice()),
            _marker: PhantomData,
        }
    }
}

/// Iterator over the elements of a [`FlatVec`].
///
/// Returned by [`FlatVec::iter`].
pub struct FlatVecIter<'buf, T> {
    /// Cursor over the vector's in-use bytes, positioned at the start of the
    /// next element to decode.
    reader: FlatReader<'buf>,
    _marker: PhantomData<T>,
}

impl<'buf, T: FlatRead<'buf>> Iterator for FlatVecIter<'buf, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        T::read_from(&mut self.reader).ok()
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

    impl FlatWrite for Point {
        fn write_into(&self, writer: &mut FlatWriter) -> Result<(), InsufficientSpace> {
            unsafe {
                writer.write(self.val)?;
                writer.write(self.ts)?;
            }
            Ok(())
        }
    }

    impl<'buf> FlatRead<'buf> for Point {
        fn read_from(reader: &mut FlatReader<'buf>) -> Result<Self, InsufficientSpace> {
            unsafe {
                let val = reader.read::<f64>()?;
                let ts = reader.read::<u64>()?;
                Ok(Self { val, ts })
            }
        }
    }

    #[test]
    fn test_push_and_first() {
        let mut vec = FlatVec::<Point>::new();
        let point = Point { val: 3.14, ts: 255 };
        vec.push(point);
        let retrieved = vec.first().unwrap();
        assert_eq!(retrieved, point);
    }

    #[test]
    fn test_iterator() {
        let mut flatvec = FlatVec::<Point>::new();
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
        let vec = FlatVec::<Point>::new();
        let collected: Vec<Point> = vec.iter().collect();
        assert_eq!(collected.len(), 0);
    }

    /// variable-sized example
    #[derive(Debug, PartialEq, Clone)]
    struct Tag<'buf> {
        val: &'buf str,
        ts: f64,
    }

    impl FlatWrite for Tag<'_> {
        fn write_into(&self, writer: &mut FlatWriter) -> Result<(), InsufficientSpace> {
            unsafe {
                // Write the string length first
                writer.write(self.val.len())?;

                // Write the string byte by byte
                for byte in self.val.bytes() {
                    writer.write(byte)?;
                }

                // Write the timestamp
                writer.write(self.ts)?;
            }
            Ok(())
        }
    }

    impl<'buf> FlatRead<'buf> for Tag<'buf> {
        fn read_from(reader: &mut FlatReader<'buf>) -> Result<Self, InsufficientSpace> {
            unsafe {
                // Read the string length
                let len = reader.read::<usize>()?;
                // Convert the bytes to a string
                let bytes = reader.read_slice::<u8>(len)?;
                let val = str::from_utf8_unchecked(bytes);

                // Read the timestamp
                let ts = reader.read::<f64>()?;

                Ok(Self { val, ts })
            }
        }
    }

    #[test]
    fn test_tag_with_string() {
        let mut flatvec = FlatVec::<Tag>::new();
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
}
