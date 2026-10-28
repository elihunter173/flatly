mod align;
mod read;
mod write;

#[cfg(test)]
mod tests;

pub use align::AlignOfVisitor;
pub use read::ReadVisitor;
pub use write::WriteVisitor;

/// Traverses fields whose borrowed data has lifetime `'buf`.
///
/// Readers may create references valid for `'buf`; writers and alignment visitors
/// support any field lifetime independently of their own backing buffers.
pub trait FlatVisitor<'buf> {
    /// Copies a scalar without retaining non-static source borrows.
    ///
    /// ```compile_fail
    /// use flatly::visitor::{FlatError, FlatVisitor};
    /// unsafe fn borrowed_scalar<'buf, V: FlatVisitor<'buf>>(
    ///     visitor: &mut V, place: *mut &'buf u64,
    /// ) -> Result<(), FlatError> {
    ///     unsafe { visitor.scalar(place) }
    /// }
    /// ```
    unsafe fn scalar<T: Copy + 'static>(&mut self, place: *mut T) -> Result<(), FlatError>;

    /// Visits a length-prefixed slice without exposing its length to the schema.
    ///
    /// The length is encoded as `L`. Readers initialize the slice reference;
    /// writers copy its elements without retaining non-static source borrows.
    /// Alignment visitors account for both `L` and `T`, even for empty slices.
    ///
    /// The slice itself may still borrow the buffer for `'buf`.
    ///
    /// ```compile_fail
    /// use flatly::visitor::{FlatError, FlatVisitor};
    /// unsafe fn borrowed_elements<'buf, V: FlatVisitor<'buf>>(
    ///     visitor: &mut V, place: *mut &'buf [&'buf u64],
    /// ) -> Result<(), FlatError> {
    ///     unsafe { visitor.slice::<u32, _>(place) }
    /// }
    /// ```
    unsafe fn slice<L: FlatLen, T: Copy + 'static>(
        &mut self,
        place: *mut &'buf [T],
    ) -> Result<(), FlatError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatError {
    InsufficientSpace,
    LenOverflow,
}

pub trait FlatLen: Copy + 'static + sealed::Sealed {
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
