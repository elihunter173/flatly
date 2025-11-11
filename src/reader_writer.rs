use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

pub trait FlatWrite {
    fn write_into(&self, w: &mut FlatWriter) -> Result<(), InsufficientSpace>;
}

pub trait FlatRead<'buf>: Sized {
    fn read_from(r: &mut FlatReader<'buf>) -> Result<Self, InsufficientSpace>;
}

// TODO: I think I should have a FlatWriter enum, one that is an "accounting" writer and one that
// is the actual writer
pub struct FlatWriter<'buf> {
    start: NonNull<u8>,
    end: NonNull<u8>,
    _lifetime: PhantomData<&'buf [u8]>,
}

// Error impl
#[derive(Debug)]
pub struct InsufficientSpace;

impl<'buf> FlatWriter<'buf> {
    pub(crate) fn from_uninit_bytes(bytes: &'buf mut [MaybeUninit<u8>]) -> Self {
        let range = bytes.as_mut_ptr_range();
        Self {
            start: NonNull::new(range.start.cast::<u8>()).unwrap(),
            end: NonNull::new(range.end.cast::<u8>()).unwrap(),
            _lifetime: PhantomData,
        }
    }

    pub unsafe fn write<T>(&mut self, v: T) -> Result<(), InsufficientSpace> {
        self.align_to(align_of::<T>())?;
        let to_write = size_of::<T>();
        if self.start.addr().get() + to_write > self.end.addr().get() {
            return Err(InsufficientSpace);
        }
        unsafe {
            self.start.cast::<T>().write(v);
            self.start = self.start.add(to_write);
            Ok(())
        }
    }

    pub fn align_to(&mut self, align: usize) -> Result<(), InsufficientSpace> {
        let offset = self.start.align_offset(align);
        if self.start.addr().get() + offset > self.end.addr().get() {
            return Err(InsufficientSpace);
        }
        unsafe {
            self.start = self.start.add(offset);
        }
        Ok(())
    }

    pub(crate) fn as_ptr(&self) -> NonNull<u8> {
        self.start
    }
}

pub struct FlatReader<'buf> {
    start: NonNull<u8>,
    end: NonNull<u8>,
    _lifetime: PhantomData<&'buf [u8]>,
}

impl<'buf> FlatReader<'buf> {
    pub(crate) fn from_bytes(bytes: &'buf [u8]) -> Self {
        let range = bytes.as_ptr_range();
        Self {
            start: NonNull::new(range.start.cast_mut().cast::<u8>()).unwrap(),
            end: NonNull::new(range.end.cast_mut().cast::<u8>()).unwrap(),
            _lifetime: PhantomData,
        }
    }

    pub unsafe fn read<T>(&mut self) -> Result<T, InsufficientSpace> {
        self.align_to(align_of::<T>())?;
        let to_read = size_of::<T>();
        if self.start.addr().get() + to_read > self.end.addr().get() {
            return Err(InsufficientSpace);
        }
        unsafe {
            let v = self.start.cast::<T>().read();
            self.start = self.start.add(to_read);
            Ok(v)
        }
    }

    pub fn align_to(&mut self, align: usize) -> Result<(), InsufficientSpace> {
        self.advance(self.start.align_offset(align))
    }

    pub fn advance(&mut self, offset: usize) -> Result<(), InsufficientSpace> {
        if self.start.addr().get() + offset > self.end.addr().get() {
            return Err(InsufficientSpace);
        }
        unsafe {
            self.start = self.start.add(offset);
        }
        Ok(())
    }

    pub(crate) fn as_ptr(&self) -> NonNull<u8> {
        self.start
    }
}
