module feature_stage_stub (
    input wire clk,
    input wire rst,
    input wire start,
    output reg done
);
    always @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
        end else if (start) begin
            done <= 1'b1;
        end else begin
            done <= 1'b0;
        end
    end
endmodule
