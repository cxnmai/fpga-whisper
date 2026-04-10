`timescale 1ns/1ps
`include "gemm_tile_vectors.vh"

module gemm_tile_i16x8_tb;
    wire signed [TILE_ROWS * TILE_COLS * 64 - 1:0] result_tile;
    integer result_file;
    integer idx;

    gemm_tile_accum_i16x8 #(
        .ROWS(TILE_ROWS),
        .COLS(TILE_COLS)
    ) dut (
        .lhs_tile(LHS_TILE),
        .rhs_tile(RHS_TILE),
        .accum_tile(ACCUM_TILE),
        .result_tile(result_tile)
    );

    function signed [63:0] result_at;
        input integer index;
        begin
            result_at = result_tile[(index * 64) +: 64];
        end
    endfunction

    initial begin
        result_file = $fopen("gemm_tile_result.txt", "w");
        if (result_file == 0) begin
            $display("failed to open gemm_tile_result.txt");
            $finish;
        end

        #1;
        for (idx = 0; idx < TILE_ROWS * TILE_COLS; idx = idx + 1) begin
            $fdisplay(result_file, "%0d", result_at(idx));
        end
        $fclose(result_file);
        $dumpfile("gemm_tile_i16x8_tb.vcd");
        $dumpvars(0, gemm_tile_i16x8_tb);
        #1;
        $finish;
    end
endmodule
