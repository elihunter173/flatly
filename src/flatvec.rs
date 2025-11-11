use std::{cmp::max, marker::PhantomData};

use crate::{FlatReader, FlatValue, FlatWriter, reader_writer::InsufficientSpace};

pub struct FlatVec<T> {
    data: Vec<u8>,
    _marker: PhantomData<T>,
}

impl<T: FlatValue> FlatVec<T> {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn push(&mut self, v: T) {
        loop {
            let mut writer = FlatWriter::from_uninit_bytes(self.data.spare_capacity_mut());
            match v.write_into(&mut writer) {
                Ok(()) => {
                    let write_addr = writer.write_head().addr().get();
                    let new_len = self.data.as_ptr().addr() - write_addr;
                    unsafe { self.data.set_len(new_len) };
                }
                Err(InsufficientSpace) => {
                    println!("growing");
                    self.data.reserve(max(8, self.data.capacity()));
                }
            }
        }
    }

    fn first(&self) -> T {
        let mut reader = FlatReader::from_bytes(self.data.as_slice());
        T::read_from(&mut reader).unwrap()
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
            writer.write(self.val)?;
            writer.write(self.ts)?;
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
        let point = Point {
            val: 3.14,
            ts: 1234567890,
        };
        println!("hi");
        vec.push(point);
        let retrieved = vec.first();
        assert_eq!(retrieved, point);
    }
}
