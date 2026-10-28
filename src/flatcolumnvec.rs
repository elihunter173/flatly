//! Columnar storage for flat, variable-sized values.

use std::{
    alloc::{self, Layout},
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
    ptr::{self, NonNull},
    slice,
};

use crate::visitor::{
    AlignOfVisitor, FlatError, FlatVisitor, MeasureVisitor, ReadVisitor, WriteVisitor,
};

/// Defines the columns and encoding of a row.
///
/// # Safety
///
/// Successful traversals must initialize a valid value and use the same sequence of
/// operations, types, and column indices, independent of the value, lifetime, and
/// visitor implementation. Traversals may terminate early on visitor errors.
/// `Columns<T>` must be a fixed-size array-like collection with the same length
/// and ordering for every `T`; `from_fn`, `as_ref`, and `as_mut` must agree on those
/// elements. Alignment discovery visits uninitialized value storage.
pub unsafe trait FlatColumnVecSchema {
    /// The value type, which may borrow from the encoded buffer for `'buf`.
    ///
    /// No `Self: 'buf` bound is needed: values borrow the buffer, not the schema.
    type Value<'buf>;

    type Columns<T>: Array<T>;

    /// Visits every field in the encoded representation without reading from `place` directly.
    ///
    /// Implementations may form raw pointers to fields and pass them to the visitor, but must not
    /// dereference `place` themselves, including after a visitor operation.
    ///
    /// # Safety
    ///
    /// `place` must be valid for the visitor's operations: writable storage for
    /// a reader or alignment visitor, or an initialized value for a writer or
    /// measurement visitor.
    unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
        place: *mut Self::Value<'buf>,
        v: &mut Self::Columns<V>,
    ) -> Result<(), FlatError>;
}

/// A fixed-size, ordered collection supporting direct shared and mutable indexing.
///
/// Sealed to built-in arrays so downstream collections cannot violate the column
/// storage invariants.
///
/// ```compile_fail,E0277
/// use flatly::flatcolumnvec::Array;
/// struct CustomColumns<T>(Vec<T>);
/// impl<T> AsRef<[T]> for CustomColumns<T> {
///     fn as_ref(&self) -> &[T] { &self.0 }
/// }
/// impl<T> AsMut<[T]> for CustomColumns<T> {
///     fn as_mut(&mut self) -> &mut [T] { &mut self.0 }
/// }
/// impl<T> std::ops::Index<usize> for CustomColumns<T> {
///     type Output = T;
///     fn index(&self, i: usize) -> &T { &self.0[i] }
/// }
/// impl<T> std::ops::IndexMut<usize> for CustomColumns<T> {
///     fn index_mut(&mut self, i: usize) -> &mut T { &mut self.0[i] }
/// }
/// impl<T> Array<T> for CustomColumns<T> {
///     fn from_fn(mut f: impl FnMut(usize) -> T) -> Self {
///         Self(vec![f(0)])
///     }
/// }
/// ```
pub trait Array<T>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize> + crate::sealed::Sealed
{
    fn from_fn(f: impl FnMut(usize) -> T) -> Self;
}

impl<T, const N: usize> crate::sealed::Sealed for [T; N] {}

impl<T, const N: usize> Array<T> for [T; N] {
    fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        std::array::from_fn(f)
    }
}

/// A vector of rows encoded into independently allocated column buffers.
///
/// Borrowed inputs are copied. Decoded references borrow this vector, preventing
/// mutation or deallocation while they are in use. Zero-byte columns and rows
/// are supported because the vector tracks its row count separately.
///
/// Decoded values cannot escape the vector:
///
/// ```compile_fail,E0515
/// use flatly::flatcolumnvec::{FlatColumnVec, FlatColumnVecSchema};
/// fn escape<'buf, S: FlatColumnVecSchema + 'buf>() -> S::Value<'buf> {
///     let values = FlatColumnVec::<S>::new();
///     values.first().unwrap()
/// }
/// ```
///
/// Mutation cannot invalidate a live decoded value:
///
/// ```compile_fail,E0502
/// use flatly::flatcolumnvec::{FlatColumnVec, FlatColumnVecSchema};
/// fn mutate<'buf, S: FlatColumnVecSchema>(
///     values: &'buf mut FlatColumnVec<S>, next: S::Value<'_>,
/// ) -> S::Value<'buf> {
///     let first = values.first().unwrap();
///     values.push(next);
///     first
/// }
/// ```
pub struct FlatColumnVec<S: FlatColumnVecSchema> {
    columns: S::Columns<Column>,
    len: usize,
}

// Each column owns an allocation; FlatColumnVec frees it using the schema's
// recomputed alignment. Bytes below len contain committed encodings, including
// potentially uninitialized padding. Bytes after len are spare storage.
struct Column {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

impl Column {
    fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    fn as_slice(&self) -> &[MaybeUninit<u8>] {
        if self.capacity == 0 {
            &[]
        } else {
            // SAFETY: the allocation covers len bytes; MaybeUninit allows padding.
            unsafe { slice::from_raw_parts(self.ptr.as_ptr().cast(), self.len) }
        }
    }

    fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        if self.capacity == 0 {
            &mut []
        } else {
            // SAFETY: len <= capacity and the exclusive borrow covers spare storage.
            unsafe {
                slice::from_raw_parts_mut(
                    self.ptr.as_ptr().add(self.len).cast(),
                    self.capacity - self.len,
                )
            }
        }
    }

    fn reserve(&mut self, required: usize, alignment: usize) {
        if required <= self.capacity && (self.capacity != 0 || alignment == 1) {
            return;
        }
        // Even a zero-byte column needs an aligned allocation if its visitor
        // operations require alignment greater than that of the dangling pointer.
        let capacity = self.capacity.saturating_mul(2).max(required).max(8);
        let layout = Layout::from_size_align(capacity, alignment).expect("invalid layout");
        // SAFETY: alignment is stable for this column; realloc preserves its
        // initialized bytes, padding, and alignment. A failed allocation leaves
        // the previous allocation owned by this column.
        let ptr = unsafe {
            if self.capacity == 0 {
                alloc::alloc(layout)
            } else {
                let old_layout =
                    Layout::from_size_align(self.capacity, alignment).expect("invalid layout");
                alloc::realloc(self.ptr.as_ptr(), old_layout, capacity)
            }
        };
        self.ptr = NonNull::new(ptr).expect("allocation failed");
        self.capacity = capacity;
    }
}

impl<S: FlatColumnVecSchema> Default for FlatColumnVec<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: FlatColumnVecSchema> FlatColumnVec<S> {
    /// Creates an empty vector without allocating any column buffers.
    pub fn new() -> Self {
        Self {
            columns: S::Columns::from_fn(|_| Column::new()),
            len: 0,
        }
    }

    fn alignments() -> S::Columns<AlignOfVisitor> {
        let mut visitors = S::Columns::from_fn(|_| AlignOfVisitor::new());
        let mut place = MaybeUninit::<S::Value<'_>>::uninit();
        // SAFETY: alignment visitation only inspects field types, not storage.
        unsafe { S::visit(place.as_mut_ptr(), &mut visitors) }
            .expect("alignment visitor never returns errors");
        visitors
    }

    /// Measures the additional bytes needed per column at the current offsets.
    fn measure(&self, value: &S::Value<'_>) -> S::Columns<usize> {
        // Column allocation bases are aligned for every encoded field type.
        let mut measures =
            S::Columns::from_fn(|i| MeasureVisitor::from_offset(self.columns[i].len));
        // SAFETY: value is initialized. Measurement only reads slice lengths;
        // it never writes through the pointer derived from this shared reference.
        match unsafe { S::visit(ptr::from_ref(value).cast_mut(), &mut measures) } {
            Ok(()) => {}
            Err(FlatError::InsufficientSpace) => panic!("encoded size overflow"),
            Err(FlatError::LenOverflow) => {
                panic!("slice length doesn't fit in its stored length type");
            }
        }
        S::Columns::from_fn(|i| measures[i].size())
    }

    /// Returns the number of rows, including rows with zero-byte encodings.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Measures a row, reserves each column's required capacity, and writes once.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails, capacity or row count overflows, or a slice's
    /// length does not fit its encoded length type. Measurement failures leave
    /// the vector unchanged. A failed append never commits a partial row, though
    /// allocation or write failures may grow buffers or initialize spare storage.
    pub fn push(&mut self, mut value: S::Value<'_>) {
        let new_len = self.len.checked_add(1).expect("row count overflow");
        let alignments = Self::alignments();
        if cfg!(debug_assertions) {
            let mut actual = S::Columns::from_fn(|_| AlignOfVisitor::new());
            // SAFETY: value is initialized; alignment visitors only inspect types.
            unsafe { S::visit(&raw mut value, &mut actual) }
                .expect("alignment visitor never returns errors");
            debug_assert!(
                alignments
                    .as_ref()
                    .iter()
                    .map(AlignOfVisitor::alignment)
                    .eq(actual.as_ref().iter().map(AlignOfVisitor::alignment)),
                "FlatColumnVec schema alignment changed between traversals"
            );
        }

        let sizes = self.measure(&value);
        let lengths = S::Columns::from_fn(|i| self.columns[i].len + sizes[i]);
        for ((column, &required), alignment) in self
            .columns
            .as_mut()
            .iter_mut()
            .zip(lengths.as_ref())
            .zip(alignments.as_ref())
        {
            column.reserve(required, alignment.alignment());
        }

        // We can't do self.columns[i] here because otherwise each iteration would try to borrow
        // self.columns for itself
        let mut columns = self.columns.as_mut().iter_mut();
        let mut writers = S::Columns::from_fn(|_| {
            let column = columns.next().expect("column count changed");
            WriteVisitor::from_uninit_bytes(column.spare_capacity_mut())
        });
        // SAFETY: value is initialized; writers copy into disjoint spare column
        // storage sized by the successful measurement traversal.
        unsafe { S::visit(&raw mut value, &mut writers) }
            .expect("FlatColumnVec schema write failed after successful measurement");
        drop(writers);

        for (column, &length) in self.columns.as_mut().iter_mut().zip(lengths.as_ref()) {
            column.len = length;
        }
        self.len = new_len;
    }

    pub fn first(&self) -> Option<S::Value<'_>> {
        self.iter().next()
    }

    /// Iterates over rows; returned references borrow the vector's column buffers.
    pub fn iter(&self) -> FlatColumnVecIter<'_, S> {
        FlatColumnVecIter {
            readers: S::Columns::from_fn(|i| {
                ReadVisitor::from_uninit_bytes(self.columns[i].as_slice())
            }),
            remaining: self.len,
            _marker: PhantomData,
        }
    }
}

impl<S: FlatColumnVecSchema> Drop for FlatColumnVec<S> {
    fn drop(&mut self) {
        if self
            .columns
            .as_ref()
            .iter()
            .all(|column| column.capacity == 0)
        {
            return;
        }
        let alignments = Self::alignments();
        for (column, alignment) in self.columns.as_mut().iter_mut().zip(alignments.as_ref()) {
            if column.capacity != 0 {
                let layout = Layout::from_size_align(column.capacity, alignment.alignment())
                    .expect("invalid layout");
                // SAFETY: this allocation is live and layout matches allocation/growth.
                unsafe { alloc::dealloc(column.ptr.as_ptr(), layout) };
            }
        }
    }
}

/// Iterator over the rows of a [`FlatColumnVec`].
pub struct FlatColumnVecIter<'buf, S: FlatColumnVecSchema> {
    readers: S::Columns<ReadVisitor<'buf>>,
    remaining: usize,
    _marker: PhantomData<&'buf FlatColumnVec<S>>,
}

impl<'buf, S: FlatColumnVecSchema> Iterator for FlatColumnVecIter<'buf, S> {
    type Item = S::Value<'buf>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let mut out = MaybeUninit::<S::Value<'buf>>::uninit();
        // SAFETY: columns contain committed rows of S. A successful traversal
        // initializes every field and borrows only from these readers' buffers.
        unsafe {
            if let Err(error) = S::visit(out.as_mut_ptr(), &mut self.readers) {
                self.remaining = 0;
                panic!("failed to read committed FlatColumnVec row: {error:?}");
            }
            self.remaining -= 1;
            Some(out.assume_init())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<S: FlatColumnVecSchema> ExactSizeIterator for FlatColumnVecIter<'_, S> {}
impl<S: FlatColumnVecSchema> std::iter::FusedIterator for FlatColumnVecIter<'_, S> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(64))]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Aligned {
        prefix: u8,
        value: u64,
    }

    #[derive(Debug, PartialEq)]
    struct Row<'buf> {
        prefix: u8,
        id: u64,
        name: &'buf [u8],
        values: &'buf [Aligned],
        unit: (),
    }

    struct RowSchema;

    unsafe impl FlatColumnVecSchema for RowSchema {
        type Value<'buf> = Row<'buf>;
        type Columns<T> = [T; 5];

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Row<'buf>,
            v: &mut [V; 5],
        ) -> Result<(), FlatError> {
            unsafe {
                v[0].scalar(&raw mut (*place).prefix)?;
                v[1].slice::<u16, _>(&raw mut (*place).name)?;
                v[0].scalar(&raw mut (*place).id)?;
                v[2].slice::<u8, _>(&raw mut (*place).values)?;
                v[3].scalar(&raw mut (*place).unit)?;
            }
            // Column 4 is intentionally unused.
            Ok(())
        }
    }

    #[test]
    fn arrays_are_initialized_in_order_including_non_copy_elements() {
        let values: [String; 3] = Array::from_fn(|i| i.to_string());
        assert_eq!(values, ["0", "1", "2"]);
        let empty: [String; 0] = Array::from_fn(|_| panic!("empty array called initializer"));
        assert!(empty.is_empty());
    }

    #[test]
    fn measures_shared_values_at_current_column_offsets() {
        let mut values = FlatColumnVec::<RowSchema>::new();
        let row = Row {
            prefix: 1,
            id: 10,
            name: b"a",
            values: &[],
            unit: (),
        };
        assert_eq!(values.measure(&row), [16, 3, 64, 0, 0]);
        assert!(values.is_empty());
        assert!(values.columns.iter().all(|column| column.capacity == 0));

        values.push(Row {
            prefix: 2,
            id: 20,
            name: b"a",
            values: &[],
            unit: (),
        });
        let state = values
            .columns
            .each_ref()
            .map(|column| (column.ptr, column.len, column.capacity));
        assert_eq!(values.measure(&row), [16, 4, 64, 0, 0]);
        assert_eq!(
            values
                .columns
                .each_ref()
                .map(|column| (column.ptr, column.len, column.capacity)),
            state
        );
        assert_eq!(values.len(), 1);
        assert_eq!(
            row,
            Row {
                prefix: 1,
                id: 10,
                name: b"a",
                values: &[],
                unit: ()
            }
        );
        assert_eq!(FlatColumnVec::<UnitSchema>::new().measure(&()), []);
    }

    #[test]
    fn empty_vector_and_exact_size_iteration() {
        let values = FlatColumnVec::<RowSchema>::default();
        assert!(values.is_empty());
        assert_eq!(values.len(), 0);
        assert_eq!(values.first(), None);
        let mut iter = values.iter();
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert!(values.columns.iter().all(|column| column.capacity == 0));
    }

    #[test]
    fn borrowed_rows_padding_and_alignment_survive_growth() {
        let mut values = FlatColumnVec::<RowSchema>::new();
        for i in 0..30 {
            // Both input buffers are dropped at the end of each iteration.
            let name = vec![i as u8; i * 11];
            let elements = vec![
                Aligned {
                    prefix: i as u8,
                    value: i as u64
                };
                i % 4
            ];
            values.push(Row {
                prefix: i as u8,
                id: i as u64 * 100,
                name: &name,
                values: &elements,
                unit: (),
            });
            let first = values.first().unwrap();
            assert_eq!(first.id, 0);
            assert!(first.name.is_empty());
            assert!(first.values.is_empty());
        }
        assert_eq!(values.len(), 30);
        assert!(!values.is_empty());
        assert_eq!(values.columns[2].ptr.as_ptr().addr() % 64, 0);
        // Zero-byte and unused columns do not grow with the others.
        assert_eq!(values.columns[3].capacity, 0);
        assert_eq!(values.columns[4].capacity, 0);

        let mut iter = values.iter();
        for i in 0..30 {
            assert_eq!(iter.len(), 30 - i);
            assert_eq!(iter.size_hint(), (30 - i, Some(30 - i)));
            let row = iter.next().unwrap();
            assert_eq!(row.prefix, i as u8);
            assert_eq!(row.id, i as u64 * 100);
            assert_eq!(row.name, vec![i as u8; i * 11]);
            assert_eq!(
                row.values,
                vec![
                    Aligned {
                        prefix: i as u8,
                        value: i as u64
                    };
                    i % 4
                ]
            );
            assert_eq!(row.values.as_ptr().addr() % 64, 0);
        }
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn only_columns_exceeding_capacity_grow() {
        let mut values = FlatColumnVec::<RowSchema>::new();
        values.push(Row {
            prefix: 1,
            id: 1,
            name: b"a",
            values: &[],
            unit: (),
        });
        let capacities = values.columns.each_ref().map(|column| column.capacity);
        values.push(Row {
            prefix: 2,
            id: 2,
            name: &vec![b'b'; 4096],
            values: &[],
            unit: (),
        });
        assert_eq!(values.columns[0].capacity, capacities[0] * 2);
        assert!(values.columns[1].capacity >= 4096);
        // The aligned column only grows enough to accommodate its next empty slice.
        assert_eq!(values.columns[2].capacity, capacities[2] * 2);
        assert_eq!(values.columns[3].capacity, 0);
        assert_eq!(values.columns[4].capacity, 0);
        assert_eq!(
            values.iter().map(|row| row.name.len()).collect::<Vec<_>>(),
            [1, 4096]
        );
    }

    struct ShortSliceSchema;

    unsafe impl FlatColumnVecSchema for ShortSliceSchema {
        type Value<'buf> = (u64, &'buf [u8]);
        type Columns<T> = [T; 2];

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut Self::Value<'buf>,
            v: &mut [V; 2],
        ) -> Result<(), FlatError> {
            unsafe {
                v[0].scalar(&raw mut (*place).0)?;
                v[1].slice::<u8, _>(&raw mut (*place).1)
            }
        }
    }

    #[test]
    fn length_overflow_does_not_commit_a_partial_row() {
        let mut values = FlatColumnVec::<ShortSliceSchema>::new();
        values.push((1, b"first"));
        let state = values
            .columns
            .each_ref()
            .map(|column| (column.ptr, column.len, column.capacity));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values.push((999, &[0; 300]));
        }));
        assert!(result.is_err());
        assert_eq!(values.len(), 1);
        assert_eq!(
            values
                .columns
                .each_ref()
                .map(|column| (column.ptr, column.len, column.capacity)),
            state
        );
        assert_eq!(values.first(), Some((1, &b"first"[..])));
        values.push((2, b"second"));
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [(1, &b"first"[..]), (2, &b"second"[..])]
        );
    }

    #[test]
    fn invalid_first_append_does_not_allocate_any_columns() {
        let mut values = FlatColumnVec::<ShortSliceSchema>::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values.push((999, &[0; 300]));
        }));
        assert!(result.is_err());
        assert!(values.is_empty());
        assert!(
            values
                .columns
                .iter()
                .all(|column| column.capacity == 0 && column.len == 0)
        );
        values.push((1, b"valid"));
        assert_eq!(values.first(), Some((1, &b"valid"[..])));
    }

    #[test]
    fn measured_appends_use_one_write_traversal_and_preserve_column_offsets() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static WRITES: AtomicUsize = AtomicUsize::new(0);
        struct CountedSchema;

        unsafe impl FlatColumnVecSchema for CountedSchema {
            type Value<'buf> = Row<'buf>;
            type Columns<T> = [T; 5];

            unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
                place: *mut Row<'buf>,
                v: &mut [V; 5],
            ) -> Result<(), FlatError> {
                if std::any::type_name::<V>() == std::any::type_name::<WriteVisitor<'_>>() {
                    WRITES.fetch_add(1, Ordering::Relaxed);
                }
                // SAFETY: forwards the same value storage and column visitors.
                unsafe { RowSchema::visit(place, v) }
            }
        }

        WRITES.store(0, Ordering::Relaxed);
        let mut values = FlatColumnVec::<CountedSchema>::new();
        values.push(Row {
            prefix: 1,
            id: 10,
            name: b"a",
            values: &[],
            unit: (),
        });
        assert_eq!(WRITES.load(Ordering::Relaxed), 1);
        assert_eq!(values.columns[1].len, 3);

        let large = vec![42; 32_768];
        values.push(Row {
            prefix: 2,
            id: 20,
            name: &large,
            values: &[],
            unit: (),
        });
        assert_eq!(WRITES.load(Ordering::Relaxed), 2);
        // The second u16 prefix needs padding after the first odd-length encoding.
        assert_eq!(values.columns[1].len, 3 + 1 + 2 + large.len());
        assert_eq!(values.columns[1].capacity, values.columns[1].len);
        values.push(Row {
            prefix: 3,
            id: 30,
            name: b"end",
            values: &[],
            unit: (),
        });
        assert_eq!(WRITES.load(Ordering::Relaxed), 3);

        let rows = values.iter().collect::<Vec<_>>();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].name, b"a");
        assert_eq!(rows[1].name, large);
        assert_eq!(rows[2].name, b"end");
        assert_eq!(
            rows.iter().map(|row| row.id).collect::<Vec<_>>(),
            [10, 20, 30]
        );
        assert_eq!(values.columns[3].capacity, 0);
        assert_eq!(values.columns[4].capacity, 0);
    }

    struct UnitSchema;

    unsafe impl FlatColumnVecSchema for UnitSchema {
        type Value<'buf> = ();
        type Columns<T> = [T; 0];

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            _place: *mut (),
            _v: &mut [V; 0],
        ) -> Result<(), FlatError> {
            // () is valid even without writing any bytes.
            Ok(())
        }
    }

    #[test]
    fn zero_columns_and_zero_byte_rows_are_counted() {
        let mut values = FlatColumnVec::<UnitSchema>::new();
        for _ in 0..10 {
            values.push(());
        }
        assert_eq!(values.len(), 10);
        assert_eq!(values.first(), Some(()));
        assert_eq!(values.iter().collect::<Vec<_>>(), vec![(); 10]);
    }

    struct AlignedUnitSchema;

    unsafe impl FlatColumnVecSchema for AlignedUnitSchema {
        type Value<'buf> = AlignedUnit;
        type Columns<T> = [T; 1];

        unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
            place: *mut AlignedUnit,
            v: &mut [V; 1],
        ) -> Result<(), FlatError> {
            unsafe { v[0].scalar(place) }
        }
    }

    #[repr(align(64))]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct AlignedUnit;

    #[test]
    fn zero_byte_aligned_columns_keep_their_allocation_pointer() {
        let mut values = FlatColumnVec::<AlignedUnitSchema>::new();
        for _ in 0..3 {
            values.push(AlignedUnit);
        }
        assert_eq!(values.columns[0].len, 0);
        assert_ne!(values.columns[0].capacity, 0);
        assert_eq!(values.iter().collect::<Vec<_>>(), vec![AlignedUnit; 3]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "FlatColumnVec schema alignment changed between traversals")]
    fn alignment_is_checked_for_each_column() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static INCLUDE_ALIGNED: AtomicBool = AtomicBool::new(false);
        struct ChangingSchema;

        // Deliberately violates traversal stability to exercise the debug check.
        unsafe impl FlatColumnVecSchema for ChangingSchema {
            type Value<'buf> = (u8, Aligned);
            type Columns<T> = [T; 2];

            unsafe fn visit<'buf, V: FlatVisitor<'buf>>(
                place: *mut Self::Value<'buf>,
                v: &mut [V; 2],
            ) -> Result<(), FlatError> {
                unsafe {
                    v[0].scalar(&raw mut (*place).1)?;
                    v[1].scalar(&raw mut (*place).0)?;
                    if INCLUDE_ALIGNED.swap(true, Ordering::Relaxed) {
                        v[1].scalar(&raw mut (*place).1)?;
                    }
                }
                Ok(())
            }
        }

        INCLUDE_ALIGNED.store(false, Ordering::Relaxed);
        FlatColumnVec::<ChangingSchema>::new().push((
            0,
            Aligned {
                prefix: 0,
                value: 0,
            },
        ));
    }
}
