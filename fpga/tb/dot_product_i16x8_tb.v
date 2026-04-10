`timescale 1ns/1ps
`include "fpga/tmp/dot_product_vectors.vh"

module dot_product_i16x8_tb;
    wire signed [63:0] result;
    integer result_file;

    dot_product_i16x8 dut (
        .a0(VEC_A0),
        .a1(VEC_A1),
        .a2(VEC_A2),
        .a3(VEC_A3),
        .a4(VEC_A4),
        .a5(VEC_A5),
        .a6(VEC_A6),
        .a7(VEC_A7),
        .b0(VEC_B0),
        .b1(VEC_B1),
        .b2(VEC_B2),
        .b3(VEC_B3),
        .b4(VEC_B4),
        .b5(VEC_B5),
        .b6(VEC_B6),
        .b7(VEC_B7),
        .result(result)
    );

    initial begin
        result_file = $fopen("fpga/tmp/dot_product_result.txt", "w");
        if (result_file == 0) begin
            $display("failed to open fpga/tmp/dot_product_result.txt");
            $finish;
        end

        #1;
        $fdisplay(result_file, "%0d", result);
        $fclose(result_file);
        $dumpfile("fpga/tmp/dot_product_i16x8_tb.vcd");
        $dumpvars(0, dot_product_i16x8_tb);
        #1;
        $finish;
    end
endmodule
