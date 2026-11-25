unsafe extern "C" {
    fn mircuda_dense_vector_row_offset(row: i32, k: i32) -> i64;
    fn mircuda_dense_vector_row_blocks(n: i32) -> i32;
}

/// Row offsets are 64-bit: the last row of an accepted 262145 x 8192 shape
/// starts at element 2^31, past `i32::MAX`.
#[test]
fn row_offsets_pass_the_signed_32_bit_range() {
    let offset = |row, k| {
        // SAFETY: the native helper is a pure multiplication.
        unsafe { mircuda_dense_vector_row_offset(row, k) }
    };
    assert_eq!(offset(262_144, 8_192), 1_i64 << 31);
    assert_eq!(offset(i32::MAX, i32::MAX), i64::from(i32::MAX) * i64::from(i32::MAX));
    assert_eq!(offset(3, 999), 2_997);
}

/// The row-group grid covers every accepted `n` up to `i32::MAX` without
/// the `n + 31` overflow, and its last row, `blocks * 32 - 1`, fits `i32`.
#[test]
fn row_group_blocks_cover_every_accepted_row_count() {
    // SAFETY: the native helper is pure arithmetic.
    let blocks = |n| unsafe { mircuda_dense_vector_row_blocks(n) };
    assert_eq!(blocks(i32::MAX), 1 << 26);
    assert!(i64::from(blocks(i32::MAX)) * 32 - 1 <= i64::from(i32::MAX));
    assert_eq!([blocks(1), blocks(32), blocks(33)], [1, 1, 2]);
}
