use std::{
    alloc::{self, Layout},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::NonNull,
    slice,
};

use crate::{FlatReader, FlatValue, FlatWriter, reader_writer::InsufficientSpace};

struct Buf {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

impl Buf {
    fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    fn as_slice(&self) -> &[u8] {
        if self.capacity == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

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
            let old_layout = Layout::array::<u8>(self.capacity).expect("invalid layout");
            unsafe { alloc::realloc(self.ptr.as_ptr(), old_layout, new_capacity) }
        };

        self.ptr = NonNull::new(new_ptr).expect("allocation failed");
        self.capacity = new_capacity;
    }

    unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.capacity != 0 {
            let layout = Layout::array::<u8>(self.capacity).expect("invalid layout");
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) }
        }
    }
}

pub struct FlatVec<T> {
    data: Buf,
    _marker: PhantomData<T>,
}

impl<T: FlatValue> FlatVec<T> {
    fn new() -> Self {
        Self {
            data: Buf::new(),
            _marker: PhantomData,
        }
    }

    fn push(&mut self, v: T) {
        loop {
            let mut writer = FlatWriter::from_uninit_bytes(self.data.spare_capacity_mut());
            match v.write_into(&mut writer) {
                Ok(()) => {
                    let write_addr = writer.write_head().addr().get();
                    let new_len = write_addr - self.data.as_ptr().addr();
                    unsafe { self.data.set_len(new_len) };
                    break;
                }
                Err(InsufficientSpace) => {
                    println!("growing");
                    self.data.double();
                }
            }
        }
    }

    fn first(&self) -> Option<T> {
        let mut reader = FlatReader::from_bytes(self.data.as_slice());
        T::read_from(&mut reader).ok()
    }

    pub fn iter(&self) -> FlatVecIter<'_, T> {
        FlatVecIter {
            reader: FlatReader::from_bytes(self.data.as_slice()),
            _marker: PhantomData,
        }
    }
}

pub struct FlatVecIter<'buf, T> {
    reader: FlatReader<'buf>,
    _marker: PhantomData<T>,
}

impl<'buf, T: FlatValue> Iterator for FlatVecIter<'buf, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        T::read_from(&mut self.reader).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Point {
        val: f64,
        ts: u64,
    }

    unsafe impl FlatValue for Point {
        fn write_into(&self, writer: &mut FlatWriter) -> Result<(), InsufficientSpace> {
            unsafe {
                writer.write(self.val)?;
                writer.write(self.ts)?;
            }
            Ok(())
        }

        fn read_from(reader: &mut FlatReader) -> Result<Self, InsufficientSpace> {
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

    #[derive(Debug, PartialEq, Clone)]
    struct Tag {
        val: String,
        ts: f64,
    }

    unsafe impl FlatValue for Tag {
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

        fn read_from(reader: &mut FlatReader) -> Result<Self, InsufficientSpace> {
            unsafe {
                // Read the string length
                let len = reader.read::<usize>()?;

                // Read the string bytes
                let mut bytes = vec![0u8; len];
                for byte in &mut bytes {
                    *byte = reader.read::<u8>()?;
                }

                // Read the timestamp
                let ts = reader.read::<f64>()?;

                // Convert bytes to String
                let val = String::from_utf8_unchecked(bytes);

                Ok(Self { val, ts })
            }
        }
    }

    #[test]
    fn test_tag_with_string() {
        let mut flatvec = FlatVec::<Tag>::new();
        let tags = vec![
            Tag {
                val: String::from("hello"),
                ts: 10.0,
            },
            Tag {
                val: String::from("world"),
                ts: 20.0,
            },
            Tag {
                val: String::from("rust!"),
                ts: 30.0,
            },
        ];

        for tag in &tags {
            flatvec.push(tag.clone());
        }

        let collected: Vec<Tag> = flatvec.iter().collect();
        assert_eq!(collected, tags);
    }
}
