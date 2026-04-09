`timescale 1ns/1ps

module feature_stage_stub_tb;
    reg clk = 0;
    reg rst = 1;
    reg start = 0;
    wire done;

    feature_stage_stub dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        #20 rst = 0;
        #10 start = 1;
        #10 start = 0;
        #30 $finish;
    end

    initial begin
        $dumpfile("fpga/tmp/feature_stage_stub_tb.vcd");
        $dumpvars(0, feature_stage_stub_tb);
    end
endmodule
