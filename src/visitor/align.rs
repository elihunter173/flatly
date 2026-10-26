use std::mem::align_of;

use super::{FlatError, FlatLen, FlatVisitor};

/// Visitor that computes the largest alignment used by a `FlatVecValue` encoding.
pub struct AlignOfVisitor {
    alignment: usize,
}

impl AlignOfVisitor {
    pub fn new() -> Self {
        Self { alignment: 1 }
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    fn include<T>(&mut self) {
        self.alignment = self.alignment.max(align_of::<T>());
    }
}

impl FlatVisitor for AlignOfVisitor {
    unsafe fn scalar<T: Copy>(&mut self, _place: *mut T) -> Result<(), FlatError> {
        self.include::<T>();
        Ok(())
    }

    unsafe fn slice_len<L: FlatLen, T>(&mut self, _place: *mut &[T]) -> Result<usize, FlatError> {
        self.include::<L>();
        Ok(0)
    }

    unsafe fn slice_data<T: Copy>(
        &mut self,
        _place: *mut &[T],
        _len: usize,
    ) -> Result<(), FlatError> {
        self.include::<T>();
        Ok(())
    }
}
