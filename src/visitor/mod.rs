mod read;
mod write;

#[cfg(test)]
mod tests;

pub use read::ReadVisitor;
pub use write::WriteVisitor;

use crate::flatvec::FlatVecValue;

pub trait FlatVisitor<'buf> {
    unsafe fn scalar<T: Copy>(&mut self, place: *mut T) -> Result<(), FlatError>;

    unsafe fn slice_len<L: FlatLen, T>(
        &mut self,
        place: *mut &'buf [T],
    ) -> Result<usize, FlatError>;

    unsafe fn slice_data<T: Copy>(
        &mut self,
        place: *mut &'buf [T],
        len: usize,
    ) -> Result<(), FlatError>;

    unsafe fn nested<T: FlatVecValue<'buf>>(&mut self, place: *mut T) -> Result<(), FlatError>
    where
        Self: Sized,
    {
        unsafe { T::visit(place, self) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatError {
    InsufficientSpace,
    LenOverflow,
}

pub trait FlatLen: Copy + sealed::Sealed {
    fn from_usize(n: usize) -> Option<Self>;
    fn to_usize(self) -> Option<usize>;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! impl_flat_len {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl FlatLen for $t {
            fn from_usize(n: usize) -> Option<Self> { <$t>::try_from(n).ok() }
            fn to_usize(self) -> Option<usize> { usize::try_from(self).ok() }
        }
    )*};
}
impl_flat_len!(u8, u16, u32, u64, usize);
