module gemm_tile_accum_i16x8 #(
    parameter integer ROWS = 1,
    parameter integer COLS = 1
) (
    input signed [ROWS * 8 * 16 - 1:0] lhs_tile,
    input signed [8 * COLS * 16 - 1:0] rhs_tile,
    input signed [ROWS * COLS * 64 - 1:0] accum_tile,
    output signed [ROWS * COLS * 64 - 1:0] result_tile
);
    wire signed [ROWS * COLS * 64 - 1:0] partial_tile;

    gemm_tile_i16x8 #(
        .ROWS(ROWS),
        .COLS(COLS)
    ) partial (
        .lhs_tile(lhs_tile),
        .rhs_tile(rhs_tile),
        .result_tile(partial_tile)
    );

    genvar index;
    generate
        for (index = 0; index < ROWS * COLS; index = index + 1) begin: accum_gen
            assign result_tile[(index * 64) +: 64] =
                partial_tile[(index * 64) +: 64] + accum_tile[(index * 64) +: 64];
        end
    endgenerate
endmodule
