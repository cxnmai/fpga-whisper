module gemm_tile_i16x8 #(
    parameter integer ROWS = 1,
    parameter integer COLS = 1
) (
    input signed [ROWS * 8 * 16 - 1:0] lhs_tile,
    input signed [8 * COLS * 16 - 1:0] rhs_tile,
    output signed [ROWS * COLS * 64 - 1:0] result_tile
);
    genvar row;
    genvar col;

    generate
        for (row = 0; row < ROWS; row = row + 1) begin: row_gen
            for (col = 0; col < COLS; col = col + 1) begin: col_gen
                wire signed [63:0] dot_result;

                dot_product_i16x8 dot_inst (
                    .a0(lhs_tile[((row * 8 + 0) * 16) +: 16]),
                    .a1(lhs_tile[((row * 8 + 1) * 16) +: 16]),
                    .a2(lhs_tile[((row * 8 + 2) * 16) +: 16]),
                    .a3(lhs_tile[((row * 8 + 3) * 16) +: 16]),
                    .a4(lhs_tile[((row * 8 + 4) * 16) +: 16]),
                    .a5(lhs_tile[((row * 8 + 5) * 16) +: 16]),
                    .a6(lhs_tile[((row * 8 + 6) * 16) +: 16]),
                    .a7(lhs_tile[((row * 8 + 7) * 16) +: 16]),
                    .b0(rhs_tile[((0 * COLS + col) * 16) +: 16]),
                    .b1(rhs_tile[((1 * COLS + col) * 16) +: 16]),
                    .b2(rhs_tile[((2 * COLS + col) * 16) +: 16]),
                    .b3(rhs_tile[((3 * COLS + col) * 16) +: 16]),
                    .b4(rhs_tile[((4 * COLS + col) * 16) +: 16]),
                    .b5(rhs_tile[((5 * COLS + col) * 16) +: 16]),
                    .b6(rhs_tile[((6 * COLS + col) * 16) +: 16]),
                    .b7(rhs_tile[((7 * COLS + col) * 16) +: 16]),
                    .result(dot_result)
                );

                assign result_tile[((row * COLS + col) * 64) +: 64] = dot_result;
            end
        end
    endgenerate
endmodule
