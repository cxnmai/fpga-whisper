`timescale 1ns/1ps

module gemm_tile_i16x8_tb;
    parameter integer TILE_ROWS = 1;
    parameter integer TILE_COLS = 1;
    localparam integer TILE_INNER = 8;

    reg [15:0] lhs_mem [0:TILE_ROWS * TILE_INNER - 1];
    reg [15:0] rhs_mem [0:TILE_INNER * TILE_COLS - 1];
    reg [63:0] accum_mem [0:TILE_ROWS * TILE_COLS - 1];
    wire signed [TILE_ROWS * TILE_INNER * 16 - 1:0] lhs_tile;
    wire signed [TILE_INNER * TILE_COLS * 16 - 1:0] rhs_tile;
    wire signed [TILE_ROWS * TILE_COLS * 64 - 1:0] accum_tile;
    wire signed [TILE_ROWS * TILE_COLS * 64 - 1:0] result_tile;
    integer result_file;
    integer idx;

    genvar lane;
    generate
        for (lane = 0; lane < TILE_ROWS * TILE_INNER; lane = lane + 1) begin : pack_lhs
            assign lhs_tile[(lane * 16) +: 16] = lhs_mem[lane];
        end
        for (lane = 0; lane < TILE_INNER * TILE_COLS; lane = lane + 1) begin : pack_rhs
            assign rhs_tile[(lane * 16) +: 16] = rhs_mem[lane];
        end
        for (lane = 0; lane < TILE_ROWS * TILE_COLS; lane = lane + 1) begin : pack_accum
            assign accum_tile[(lane * 64) +: 64] = accum_mem[lane];
        end
    endgenerate

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
        $readmemh("gemm_tile_lhs.mem", lhs_mem);
        $readmemh("gemm_tile_rhs.mem", rhs_mem);
        $readmemh("gemm_tile_accum.mem", accum_mem);
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
