use std::marker::PhantomData;

use crate::{FlatValue, FlatWriter};

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
        self.data.reserve(1024);
        let mut writer = FlatWriter::from_uninit_bytes(self.data.spare_capacity_mut());
        v.write_into(&mut writer);
    }
}
