//! Flat, contiguous storage for variable sized values.
//!
//! `flatly` stores a sequence of values back-to-back in a single byte buffer instead of as an array
//! of `T`. Each value decides its own byte layout by implementing [`FlatWrite`] and [`FlatRead`].
//! This makes it possible to store variable-sized records, as a string, with no per-element heap
//! allocation. Decoded values can also borrow directly from the buffer (for example `&'buf str`).

mod flatvec;
mod reader_writer;

pub use flatvec::{FlatVec, FlatVecIter};
pub use reader_writer::{FlatRead, FlatReader, FlatWrite, FlatWriter, InsufficientSpace};
