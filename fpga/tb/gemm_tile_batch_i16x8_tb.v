`timescale 1ns/1ps

module gemm_tile_batch_i16x8_tb;
    parameter integer TILE_ROWS = 1;
    parameter integer TILE_COLS = 1;
    parameter integer CASE_COUNT = 1;
    localparam integer TILE_INNER = 8;
    localparam integer LHS_LEN = TILE_ROWS * TILE_INNER;
    localparam integer RHS_LEN = TILE_INNER * TILE_COLS;
    localparam integer ACCUM_LEN = TILE_ROWS * TILE_COLS;

    reg [15:0] lhs_mem [0:CASE_COUNT * LHS_LEN - 1];
    reg [15:0] rhs_mem [0:CASE_COUNT * RHS_LEN - 1];
    reg [63:0] accum_mem [0:CASE_COUNT * ACCUM_LEN - 1];

    reg signed [LHS_LEN * 16 - 1:0] lhs_tile;
    reg signed [RHS_LEN * 16 - 1:0] rhs_tile;
    reg signed [ACCUM_LEN * 64 - 1:0] accum_tile;
    wire signed [ACCUM_LEN * 64 - 1:0] result_tile;

    integer result_file;
    integer case_idx;
    integer lane_idx;

    gemm_tile_accum_i16x8 #(
        .ROWS(TILE_ROWS),
        .COLS(TILE_COLS)
    ) dut (
        .lhs_tile(lhs_tile),
        .rhs_tile(rhs_tile),
        .accum_tile(accum_tile),
        .result_tile(result_tile)
    );

    function signed [63:0] result_at;
        input integer index;
        begin
            result_at = result_tile[(index * 64) +: 64];
        end
    endfunction

    initial begin
        $readmemh("gemm_tile_batch_lhs.mem", lhs_mem);
        $readmemh("gemm_tile_batch_rhs.mem", rhs_mem);
        $readmemh("gemm_tile_batch_accum.mem", accum_mem);

        result_file = $fopen("gemm_tile_batch_result.txt", "w");
        if (result_file == 0) begin
            $display("failed to open gemm_tile_batch_result.txt");
            $finish;
        end

        for (case_idx = 0; case_idx < CASE_COUNT; case_idx = case_idx + 1) begin
            for (lane_idx = 0; lane_idx < LHS_LEN; lane_idx = lane_idx + 1) begin
                lhs_tile[(lane_idx * 16) +: 16] = lhs_mem[case_idx * LHS_LEN + lane_idx];
            end
            for (lane_idx = 0; lane_idx < RHS_LEN; lane_idx = lane_idx + 1) begin
                rhs_tile[(lane_idx * 16) +: 16] = rhs_mem[case_idx * RHS_LEN + lane_idx];
            end
            for (lane_idx = 0; lane_idx < ACCUM_LEN; lane_idx = lane_idx + 1) begin
                accum_tile[(lane_idx * 64) +: 64] = accum_mem[case_idx * ACCUM_LEN + lane_idx];
            end

            #1;
            for (lane_idx = 0; lane_idx < ACCUM_LEN; lane_idx = lane_idx + 1) begin
                $fdisplay(result_file, "%0d", result_at(lane_idx));
            end
        end

        $fclose(result_file);
        $dumpfile("gemm_tile_batch_i16x8_tb.vcd");
        $dumpvars(0, gemm_tile_batch_i16x8_tb);
        #1;
        $finish;
    end
endmodule
