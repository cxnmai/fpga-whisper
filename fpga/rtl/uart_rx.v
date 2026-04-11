// UART receiver -- 8N1, parameterizable baud rate.
// Two-FF synchronizer on the rx input for metastability protection.
// Samples each bit at the midpoint of the bit period.

module uart_rx #(
    parameter integer CLK_FREQ  = 100_000_000,
    parameter integer BAUD_RATE = 115_200
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] rx_data,
    output reg        rx_valid
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
    reg [7:0]  rx_shift;

    // Two-FF synchronizer
    reg rx_meta, rx_sync;
    always @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
    end

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            rx_valid <= 1'b0;
            rx_data  <= 8'd0;
            clk_cnt  <= 16'd0;
            bit_idx  <= 3'd0;
            rx_shift <= 8'd0;
        end else begin
            rx_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (!rx_sync) begin
                        state   <= S_START;
                        clk_cnt <= 16'd0;
                    end
                end

                S_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2) - 1) begin
                        if (!rx_sync) begin
                            state   <= S_DATA;
                            clk_cnt <= 16'd0;
                            bit_idx <= 3'd0;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end

                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt            <= 16'd0;
                        rx_shift[bit_idx]  <= rx_sync;
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
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        rx_valid <= 1'b1;
                        rx_data  <= rx_shift;
                        state    <= S_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
