unsafe extern "C" {
    fn mircuda_dense_vector_row_offset(row: i32, k: i32) -> i64;
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
