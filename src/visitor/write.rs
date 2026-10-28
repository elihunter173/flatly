use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

use super::{FlatError, FlatLen, FlatVisitor};

pub struct WriteVisitor<'out> {
    start: NonNull<u8>,
    end: NonNull<u8>,
    _lifetime: PhantomData<&'out mut [MaybeUninit<u8>]>,
}

impl<'out> WriteVisitor<'out> {
    pub(crate) fn from_uninit_bytes(bytes: &'out mut [MaybeUninit<u8>]) -> Self {
        let range = bytes.as_mut_ptr_range();
        Self {
            start: NonNull::new(range.start.cast()).unwrap(),
            end: NonNull::new(range.end.cast()).unwrap(),
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn as_ptr(&self) -> NonNull<u8> {
        self.start
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

// The writer copies borrowed data out, so it doesn't care what lifetime the borrows have.
impl<'buf> FlatVisitor<'buf> for WriteVisitor<'_> {
    unsafe fn scalar<T: Copy + 'static>(&mut self, place: *mut T) -> Result<(), FlatError> {
        unsafe {
            let v = place.read();
            self.take_array::<T>(1)?.write(v);
        }
        Ok(())
    }

    unsafe fn slice<L: FlatLen, T: Copy + 'static>(
        &mut self,
        place: *mut &'buf [T],
    ) -> Result<(), FlatError> {
        unsafe {
            let src = place.read();
            let mut stored = L::from_usize(src.len()).ok_or(FlatError::LenOverflow)?;
            self.scalar(&raw mut stored)?;
            let dst = self.take_array::<T>(src.len())?;
            dst.as_ptr()
                .copy_from_nonoverlapping(src.as_ptr(), src.len());
        }
        Ok(())
    }
}
