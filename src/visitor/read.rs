use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull, slice};

use super::{FlatError, FlatLen, FlatVisitor};

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

impl<'buf> FlatVisitor for ReadVisitor<'buf> {
    unsafe fn scalar<T: Copy>(&mut self, place: *mut T) -> Result<(), FlatError> {
        unsafe { place.write(self.take_array::<T>(1)?.read()) };
        Ok(())
    }

    unsafe fn slice_len<L: FlatLen, T>(&mut self, _place: *mut &[T]) -> Result<usize, FlatError> {
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
        place: *mut &[T],
        len: usize,
    ) -> Result<(), FlatError> {
        unsafe {
            let head = self.take_array::<T>(len)?;
            place.write(slice::from_raw_parts(head.as_ptr(), len));
        }
        Ok(())
    }
}
