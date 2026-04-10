`timescale 1ns/1ps
`include "gelu_vectors.vh"

module gelu_pwl_q8_8x8_tb;
    wire signed [8 * 16 - 1:0] output_block;
    integer result_file;
    integer idx;

    gelu_pwl_q8_8x8 dut (
        .input_block(INPUT_BLOCK),
        .output_block(output_block)
    );

    function signed [15:0] lane_at;
        input integer index;
        begin
            lane_at = output_block[(index * 16) +: 16];
        end
    endfunction

    initial begin
        result_file = $fopen("gelu_result.txt", "w");
        if (result_file == 0) begin
            $display("failed to open gelu_result.txt");
            $finish;
        end

        #1;
        for (idx = 0; idx < 8; idx = idx + 1) begin
            $fdisplay(result_file, "%0d", lane_at(idx));
        end
        $fclose(result_file);
        $dumpfile("gelu_pwl_q8_8x8_tb.vcd");
        $dumpvars(0, gelu_pwl_q8_8x8_tb);
        #1;
        $finish;
    end
endmodule
