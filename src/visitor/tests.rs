use std::mem::MaybeUninit;

use super::*;

#[repr(align(8))]
struct AlignedBytes<const N: usize>([MaybeUninit<u8>; N]);

#[derive(Debug, Clone, PartialEq)]
struct Row<'buf> {
    name: &'buf str,
    values: &'buf [u32],
    ts: u64,
}

unsafe fn visit_split<'buf, V: FlatVisitor<'buf>>(
    place: *mut Row<'buf>,
    names: &mut V,
    data: &mut V,
) -> Result<(), FlatError> {
    unsafe {
        let name = (&raw mut (*place).name).cast::<&'buf [u8]>();
        names.slice::<u8, _>(name)?;
        data.slice::<u16, _>(&raw mut (*place).values)?;
        data.scalar(&raw mut (*place).ts)
    }
}

#[test]
fn test_fields_in_separate_visitors() {
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

    let mut names_buf = AlignedBytes([MaybeUninit::<u8>::uninit(); 64]);
    let mut data_buf = AlignedBytes([MaybeUninit::<u8>::uninit(); 256]);
    let (names_bytes, data_bytes) = {
        let mut names = WriteVisitor::from_uninit_bytes(&mut names_buf.0);
        let mut data = WriteVisitor::from_uninit_bytes(&mut data_buf.0);
        for row in &rows {
            let mut row = row.clone();
            unsafe { visit_split(&raw mut row, &mut names, &mut data).unwrap() };
        }
        let (n, d) = (names.as_ptr().addr().get(), data.as_ptr().addr().get());
        (
            n - names_buf.0.as_ptr().addr(),
            d - data_buf.0.as_ptr().addr(),
        )
    };
    // Keep padding as MaybeUninit instead of asserting every byte is initialized.
    let names_bytes = &names_buf.0[..names_bytes];
    let data_bytes = &data_buf.0[..data_bytes];

    // Each name has its own u8 length prefix, including the empty name.
    assert_eq!(names_bytes.len(), 3 + "cpu".len() + "memory".len());

    let mut names = ReadVisitor::from_uninit_bytes(names_bytes);
    let mut data = ReadVisitor::from_uninit_bytes(data_bytes);
    let read: Vec<Row> = (0..rows.len())
        .map(|_| {
            let mut out = MaybeUninit::<Row>::uninit();
            unsafe {
                visit_split(out.as_mut_ptr(), &mut names, &mut data).unwrap();
                out.assume_init()
            }
        })
        .collect();
    assert_eq!(read, rows);
}

#[test]
fn test_slice_alignment_includes_length_and_elements_without_reading_place() {
    // Neither the slice reference nor any elements are initialized.
    let mut bytes = MaybeUninit::<&[u8]>::uninit();
    let mut visitor = AlignOfVisitor::new();
    unsafe { visitor.slice::<u64, u8>(bytes.as_mut_ptr()).unwrap() };
    assert_eq!(visitor.alignment(), align_of::<u64>());

    #[repr(align(64))]
    #[derive(Clone, Copy)]
    struct Aligned;

    let mut elements = MaybeUninit::<&[Aligned]>::uninit();
    let mut visitor = AlignOfVisitor::new();
    unsafe { visitor.slice::<u8, Aligned>(elements.as_mut_ptr()).unwrap() };
    assert_eq!(visitor.alignment(), 64);
}

#[test]
fn test_len_overflow() {
    let long = [0u8; 300];
    let mut place: &[u8] = &long;
    let mut buf = [MaybeUninit::<u8>::uninit(); 512];
    let mut w = WriteVisitor::from_uninit_bytes(&mut buf);
    let res = unsafe { w.slice::<u8, u8>(&raw mut place) };
    assert_eq!(res, Err(FlatError::LenOverflow));
    let res = unsafe { w.slice::<u16, u8>(&raw mut place) };
    assert_eq!(res, Ok(()));

    let written = w.as_ptr().addr().get() - buf.as_ptr().addr();
    let mut reader = ReadVisitor::from_uninit_bytes(&buf[..written]);
    let mut out = MaybeUninit::<&[u8]>::uninit();
    unsafe {
        reader.slice::<u16, u8>(out.as_mut_ptr()).unwrap();
        assert_eq!(out.assume_init(), &long);
    }
}
