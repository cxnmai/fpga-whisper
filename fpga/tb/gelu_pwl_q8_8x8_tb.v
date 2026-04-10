`timescale 1ns/1ps

module gelu_pwl_q8_8x8_tb;
    reg [15:0] input_mem [0:7];
    wire signed [8 * 16 - 1:0] input_block;
    wire signed [8 * 16 - 1:0] output_block;
    integer result_file;
    integer idx;

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : pack_input
            assign input_block[(lane * 16) +: 16] = input_mem[lane];
        end
    endgenerate

    gelu_pwl_q8_8x8 dut (
        .input_block(input_block),
        .output_block(output_block)
    );

    function signed [15:0] lane_at;
        input integer index;
        begin
            lane_at = output_block[(index * 16) +: 16];
        end
    endfunction

    initial begin
        $readmemh("gelu_input.mem", input_mem);
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
