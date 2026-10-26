use std::{mem::MaybeUninit, slice};

use super::*;

#[repr(align(8))]
struct AlignedBytes<const N: usize>([MaybeUninit<u8>; N]);

#[derive(Debug, Clone, PartialEq)]
struct Row<'buf> {
    name: &'buf str,
    values: &'buf [u32],
    ts: u64,
}

unsafe fn visit_split<'buf, L: FlatVisitor, D: FlatVisitor>(
    place: *mut Row<'buf>,
    lens: &mut L,
    data: &mut D,
) -> Result<(), FlatError> {
    unsafe {
        let name = (&raw mut (*place).name).cast::<&'buf [u8]>();
        let name_len = lens.slice_len::<u8, _>(name)?;
        let values_len = lens.slice_len::<u16, _>(&raw mut (*place).values)?;
        data.slice_data(name, name_len)?;
        data.slice_data(&raw mut (*place).values, values_len)?;
        data.scalar(&raw mut (*place).ts)
    }
}

#[test]
fn test_len_and_data_in_separate_visitors() {
    let rows = vec![
        Row {
            name: "cpu",
            values: &[1, 2, 3],
            ts: 10,
        },
        Row {
            name: "",
            values: &[],
            ts: 20,
        },
        Row {
            name: "memory",
            values: &[u32::MAX],
            ts: 30,
        },
    ];

    let mut lens_buf = AlignedBytes([MaybeUninit::<u8>::uninit(); 64]);
    let mut data_buf = AlignedBytes([MaybeUninit::<u8>::uninit(); 256]);
    let (lens_bytes, data_bytes) = {
        let mut lens = WriteVisitor::from_uninit_bytes(&mut lens_buf.0);
        let mut data = WriteVisitor::from_uninit_bytes(&mut data_buf.0);
        for row in &rows {
            let mut row = row.clone();
            unsafe { visit_split(&raw mut row, &mut lens, &mut data).unwrap() };
        }
        let (l, d) = (lens.as_ptr().addr().get(), data.as_ptr().addr().get());
        (
            l - lens_buf.0.as_ptr().addr(),
            d - data_buf.0.as_ptr().addr(),
        )
    };
    let lens_bytes = unsafe { slice::from_raw_parts(lens_buf.0.as_ptr().cast::<u8>(), lens_bytes) };
    let data_bytes = unsafe { slice::from_raw_parts(data_buf.0.as_ptr().cast::<u8>(), data_bytes) };

    // Lengths column is exactly (u8 name_len, pad, u16 values_len) per row.
    assert_eq!(lens_bytes.len(), 3 * 4);

    let mut lens = ReadVisitor::from_bytes(lens_bytes);
    let mut data = ReadVisitor::from_bytes(data_bytes);
    let read: Vec<Row> = (0..rows.len())
        .map(|_| {
            let mut out = MaybeUninit::<Row>::uninit();
            unsafe {
                visit_split(out.as_mut_ptr(), &mut lens, &mut data).unwrap();
                out.assume_init()
            }
        })
        .collect();
    assert_eq!(read, rows);
}

#[test]
fn test_len_overflow() {
    let long = [0u8; 300];
    let mut place: &[u8] = &long;
    let mut buf = [MaybeUninit::<u8>::uninit(); 512];
    let mut w = WriteVisitor::from_uninit_bytes(&mut buf);
    let res = unsafe { w.slice_len::<u8, u8>(&raw mut place) };
    assert_eq!(res, Err(FlatError::LenOverflow));
    let res = unsafe { w.slice_len::<u16, u8>(&raw mut place) };
    assert_eq!(res, Ok(300));
}
