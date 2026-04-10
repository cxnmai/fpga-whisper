`timescale 1ns/1ps

module dot_product_i16x8_tb;
    reg [15:0] vec_a_mem [0:7];
    reg [15:0] vec_b_mem [0:7];
    wire signed [15:0] vec_a0 = $signed(vec_a_mem[0]);
    wire signed [15:0] vec_a1 = $signed(vec_a_mem[1]);
    wire signed [15:0] vec_a2 = $signed(vec_a_mem[2]);
    wire signed [15:0] vec_a3 = $signed(vec_a_mem[3]);
    wire signed [15:0] vec_a4 = $signed(vec_a_mem[4]);
    wire signed [15:0] vec_a5 = $signed(vec_a_mem[5]);
    wire signed [15:0] vec_a6 = $signed(vec_a_mem[6]);
    wire signed [15:0] vec_a7 = $signed(vec_a_mem[7]);
    wire signed [15:0] vec_b0 = $signed(vec_b_mem[0]);
    wire signed [15:0] vec_b1 = $signed(vec_b_mem[1]);
    wire signed [15:0] vec_b2 = $signed(vec_b_mem[2]);
    wire signed [15:0] vec_b3 = $signed(vec_b_mem[3]);
    wire signed [15:0] vec_b4 = $signed(vec_b_mem[4]);
    wire signed [15:0] vec_b5 = $signed(vec_b_mem[5]);
    wire signed [15:0] vec_b6 = $signed(vec_b_mem[6]);
    wire signed [15:0] vec_b7 = $signed(vec_b_mem[7]);
    wire signed [63:0] result;
    integer result_file;

    dot_product_i16x8 dut (
        .a0(vec_a0),
        .a1(vec_a1),
        .a2(vec_a2),
        .a3(vec_a3),
        .a4(vec_a4),
        .a5(vec_a5),
        .a6(vec_a6),
        .a7(vec_a7),
        .b0(vec_b0),
        .b1(vec_b1),
        .b2(vec_b2),
        .b3(vec_b3),
        .b4(vec_b4),
        .b5(vec_b5),
        .b6(vec_b6),
        .b7(vec_b7),
        .result(result)
    );

    initial begin
        $readmemh("dot_product_a.mem", vec_a_mem);
        $readmemh("dot_product_b.mem", vec_b_mem);
        result_file = $fopen("dot_product_result.txt", "w");
        if (result_file == 0) begin
            $display("failed to open dot_product_result.txt");
            $finish;
        end

        #1;
        $fdisplay(result_file, "%0d", result);
        $fclose(result_file);
        $dumpfile("dot_product_i16x8_tb.vcd");
        $dumpvars(0, dot_product_i16x8_tb);
        #1;
        $finish;
    end
endmodule
