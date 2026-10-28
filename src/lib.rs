//! Flat, contiguous storage for variable sized values.
//!
//! `flatly` stores a sequence of values back-to-back in a single byte buffer instead of as an array
//!
//! of values. A separate schema marker type defines the byte layout by implementing
//! [`FlatVecSchema`](flatvec::FlatVecSchema). Its `Value<'buf>` associated type is what
//! callers push and read, tying decoded borrows to the backing buffer. A single visitor
//! traversal defines both reads and writes.
//!
//! This makes it possible to store variable-sized records, as a string, with no per-element heap
//! allocation. Decoded values can also borrow directly from the buffer (for example `&'buf str`).

pub mod flatvec;
pub mod visitor;
