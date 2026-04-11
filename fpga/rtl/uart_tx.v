// UART transmitter -- 8N1, parameterizable baud rate.
// Assert tx_start for one cycle while tx_busy is low to begin a byte.

module uart_tx #(
    parameter integer CLK_FREQ  = 100_000_000,
    parameter integer BAUD_RATE = 115_200
) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] tx_data,
    input  wire       tx_start,
    output reg        tx,
    output reg        tx_busy
);
    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

    localparam [1:0]
        S_IDLE  = 2'd0,
        S_START = 2'd1,
        S_DATA  = 2'd2,
        S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  tx_shift;

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            tx      <= 1'b1;
            tx_busy <= 1'b0;
            clk_cnt <= 16'd0;
            bit_idx <= 3'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx <= 1'b1;
                    if (tx_start) begin
                        state    <= S_START;
                        tx_shift <= tx_data;
                        tx_busy  <= 1'b1;
                        clk_cnt  <= 16'd0;
                    end else begin
                        tx_busy <= 1'b0;
                    end
                end

                S_START: begin
                    tx <= 1'b0;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        state   <= S_DATA;
                        clk_cnt <= 16'd0;
                        bit_idx <= 3'd0;
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end

                S_DATA: begin
                    tx <= tx_shift[bit_idx];
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
                        if (bit_idx == 3'd7) begin
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end

                S_STOP: begin
                    tx <= 1'b1;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        state   <= S_IDLE;
                        tx_busy <= 1'b0;
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
