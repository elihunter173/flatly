use std::mem::align_of;

use super::{FlatError, FlatLen, FlatVisitor};

/// Visitor that computes the largest alignment used by a `FlatVecSchema` encoding.
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

impl<'buf> FlatVisitor<'buf> for AlignOfVisitor {
    unsafe fn scalar<T: Copy + 'static>(&mut self, _place: *mut T) -> Result<(), FlatError> {
        self.include::<T>();
        Ok(())
    }

    unsafe fn slice<L: FlatLen, T: Copy + 'static>(
        &mut self,
        _place: *mut &'buf [T],
    ) -> Result<(), FlatError> {
        self.include::<L>();
        self.include::<T>();
        Ok(())
    }
}
