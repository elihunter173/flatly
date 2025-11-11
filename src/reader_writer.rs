use std::{marker::PhantomData, mem::MaybeUninit};

pub unsafe trait FlatValue {
    fn write_into(self, w: &mut FlatWriter);
    fn read_from(self, r: &mut FlatReader);
}

// TODO: I think I should have a FlatWriter enum, one that is an "accounting" writer and one that
// is the actual writer
pub struct FlatWriter<'buf> {
    start: *mut u8,
    end: *mut u8,
    _lifetime: PhantomData<&'buf [u8]>,
}

impl<'buf> FlatWriter<'buf> {
    pub(crate) fn from_uninit_bytes(bytes: &'buf mut [MaybeUninit<u8>]) -> Self {
        let range = bytes.as_mut_ptr_range();
        Self {
            start: range.start.cast::<u8>(),
            end: range.end.cast::<u8>(),
            _lifetime: PhantomData,
        }
    }

    pub fn write<T>(&mut self, v: T) {
        self.align_to(align_of::<T>());
        let to_write = size_of::<T>();
        assert!(self.start.addr() + to_write < self.end.addr());
        unsafe {
            self.start.cast::<T>().write(v);
            self.start = self.start.add(to_write);
        }
    }

    pub fn align_to(&mut self, align: usize) {
        let offset = self.start.align_offset(align);
        assert!(self.start.addr() + offset < self.end.addr());
        unsafe {
            self.start = self.start.add(offset);
        }
    }
}

pub struct FlatReader<'buf> {
    start: *const u8,
    end: *const u8,
    _lifetime: PhantomData<&'buf [u8]>,
}

impl FlatReader<'_> {
    pub fn read<T>(&mut self) -> T {
        self.align_to(align_of::<T>());
        let to_write = size_of::<T>();
        assert!(self.start.addr() + to_write < self.end.addr());
        unsafe {
            let v = self.start.cast::<T>().read();
            self.start = self.start.add(to_write);
            v
        }
    }

    pub fn align_to(&mut self, align: usize) {
        let offset = self.start.align_offset(align);
        assert!(self.start.addr() + offset < self.end.addr());
        unsafe {
            self.start = self.start.add(offset);
        }
    }
}
