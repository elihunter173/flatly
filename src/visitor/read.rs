use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull, slice};

use super::{FlatError, FlatLen, FlatVisitor};
use crate::flatvec::FlatVecValue;

pub struct ReadVisitor<'buf> {
    start: NonNull<u8>,
    end: NonNull<u8>,
    _lifetime: PhantomData<&'buf [u8]>,
}

impl<'buf> ReadVisitor<'buf> {
    pub(crate) fn from_bytes(bytes: &'buf [u8]) -> Self {
        let range = bytes.as_ptr_range();
        Self {
            start: NonNull::new(range.start.cast_mut()).unwrap(),
            end: NonNull::new(range.end.cast_mut()).unwrap(),
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn read<T: FlatVecValue<'buf>>(&mut self) -> Result<T, FlatError> {
        let mut out = MaybeUninit::<T>::uninit();
        // SAFETY: `out` is valid for writes; the `FlatVecValue` contract guarantees every field is
        // initialized on success. On error `out` is dropped as `MaybeUninit`, so nothing leaks
        // or double-drops.
        unsafe {
            T::visit(out.as_mut_ptr(), self)?;
            Ok(out.assume_init())
        }
    }

    fn take_array<T>(&mut self, len: usize) -> Result<NonNull<T>, FlatError> {
        let size = size_of::<T>()
            .checked_mul(len)
            .ok_or(FlatError::InsufficientSpace)?;
        let pad = self.start.align_offset(align_of::<T>());
        let remaining = self.end.addr().get() - self.start.addr().get();
        let needed = pad.checked_add(size).ok_or(FlatError::InsufficientSpace)?;
        if needed > remaining {
            return Err(FlatError::InsufficientSpace);
        }
        unsafe {
            let ptr = self.start.add(pad);
            self.start = ptr.add(size);
            Ok(ptr.cast())
        }
    }
}

impl<'buf> FlatVisitor<'buf> for ReadVisitor<'buf> {
    unsafe fn scalar<T: Copy>(&mut self, place: *mut T) -> Result<(), FlatError> {
        unsafe { place.write(self.take_array::<T>(1)?.read()) };
        Ok(())
    }

    unsafe fn slice_len<L: FlatLen, T>(
        &mut self,
        _place: *mut &'buf [T],
    ) -> Result<usize, FlatError> {
        let mut stored = MaybeUninit::<L>::uninit();
        unsafe {
            self.scalar(stored.as_mut_ptr())?;
            stored
                .assume_init()
                .to_usize()
                .ok_or(FlatError::LenOverflow)
        }
    }

    unsafe fn slice_data<T: Copy>(
        &mut self,
        place: *mut &'buf [T],
        len: usize,
    ) -> Result<(), FlatError> {
        unsafe {
            let head = self.take_array::<T>(len)?;
            place.write(slice::from_raw_parts(head.as_ptr(), len));
        }
        Ok(())
    }
}
