use std::marker::PhantomData;

use crate::{FlatValue, FlatWriter, reader_writer::InsufficientSpace};

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
        // TODO: Actually reserve the correct amount
        loop {
            let mut writer = FlatWriter::from_uninit_bytes(self.data.spare_capacity_mut());
            match v.write_into(&mut writer) {
                Ok(()) => break,
                Err(InsufficientSpace) => {
                    println!("growing");
                    self.data.reserve(self.data.capacity());
                }
            }
        }
    }
}
