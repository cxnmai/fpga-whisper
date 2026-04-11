// Sequential mel-filterbank + log engine for synthesis.
//
// Replaces the combinational mel_filterbank_201x80 + log_mel_frame chain
// which requires 16,080 parallel multipliers.  This version uses a single
// DSP48-inferred MAC lane and processes mel bins one at a time:
//
//   80 bins x (201 reads + 2 pipeline flush + 1 log/write) = 16,320 clocks
//   At 100 MHz: ~163 us per frame  (a Whisper frame spans 25 ms of audio).
//
// BRAMs are dual-ported: host writes via the *_wr_* ports while the engine
// reads via internal addresses during computation.

module mel_engine_seq (
    input  wire        clk,
    input  wire        rst,

    // Coefficient write port  (host -> BRAM, 16 080 x u16)
    input  wire        coeff_wr_en,
    input  wire [13:0] coeff_wr_addr,
    input  wire [15:0] coeff_wr_data,

    // Power-spectrum write port  (host -> BRAM, 201 x u24)
    input  wire        power_wr_en,
    input  wire [7:0]  power_wr_addr,
    input  wire [23:0] power_wr_data,

    // Control
    input  wire        start,
    output reg         done,
    output reg         busy,

    // Result read port  (BRAM -> host, 80 x i16)
    input  wire [6:0]  result_rd_addr,
    output reg  [15:0] result_rd_data
);

    // ----------------------------------------------------------------
    // Coefficient BRAM  (16 080 x 16-bit unsigned)
    // ----------------------------------------------------------------
    (* ram_style = "block" *) reg [15:0] coeff_mem [0:16079];
    reg [15:0] coeff_rd_reg;
    reg [13:0] coeff_rd_addr;

    always @(posedge clk) begin
        if (coeff_wr_en)
            coeff_mem[coeff_wr_addr] <= coeff_wr_data;
    end

    always @(posedge clk) begin
        coeff_rd_reg <= coeff_mem[coeff_rd_addr];
    end

    // ----------------------------------------------------------------
    // Power-spectrum BRAM  (256 x 24-bit unsigned, only 0..200 used)
    // ----------------------------------------------------------------
    reg [23:0] power_mem [0:255];
    reg [23:0] power_rd_reg;
    reg [7:0]  power_rd_addr;

    always @(posedge clk) begin
        if (power_wr_en)
            power_mem[power_wr_addr] <= power_wr_data;
    end

    always @(posedge clk) begin
        power_rd_reg <= power_mem[power_rd_addr];
    end

    // ----------------------------------------------------------------
    // Result BRAM  (128 x 16-bit, only 0..79 used)
    // ----------------------------------------------------------------
    reg [15:0] result_mem [0:127];

    always @(posedge clk) begin
        result_rd_data <= result_mem[result_rd_addr];
    end

    // ----------------------------------------------------------------
    // Log unit  (combinational -- reuses existing module)
    // ----------------------------------------------------------------
    reg  [47:0] accum;
    wire [15:0] log_result;

    log_mel_q8_8 u_log (
        .mel_value (accum),
        .log_value (log_result)
    );

    // ----------------------------------------------------------------
    // Compute FSM
    // ----------------------------------------------------------------
    localparam [1:0]
        ME_IDLE      = 2'd0,
        ME_COMPUTE   = 2'd1,
        ME_LOG_WRITE = 2'd2,
        ME_DONE      = 2'd3;

    reg [1:0]  me_state;
    reg [6:0]  mel_idx;      // 0 .. 79
    reg [8:0]  bin_idx;      // 0 .. 201 (201 = pipeline flush zone)
    reg [13:0] coeff_base;   // mel_idx * 201, maintained as running sum
    reg        rd_pipe_0;    // stage-1: read was issued
    reg        rd_pipe_1;    // stage-2: BRAM data ready for MAC

    always @(posedge clk) begin
        if (rst) begin
            me_state   <= ME_IDLE;
            done       <= 1'b0;
            busy       <= 1'b0;
            accum      <= 48'd0;
            mel_idx    <= 7'd0;
            bin_idx    <= 9'd0;
            coeff_base <= 14'd0;
            rd_pipe_0  <= 1'b0;
            rd_pipe_1  <= 1'b0;
        end else begin
            done <= 1'b0;

            case (me_state)
                // -------------------------------------------------
                ME_IDLE: begin
                    rd_pipe_0 <= 1'b0;
                    rd_pipe_1 <= 1'b0;
                    if (start) begin
                        me_state   <= ME_COMPUTE;
                        busy       <= 1'b1;
                        mel_idx    <= 7'd0;
                        bin_idx    <= 9'd0;
                        coeff_base <= 14'd0;
                        accum      <= 48'd0;
                    end
                end

                // -------------------------------------------------
                ME_COMPUTE: begin
                    // Pipeline stage 2 : MAC
                    if (rd_pipe_1)
                        accum <= accum + coeff_rd_reg * power_rd_reg;

                    // Pipeline advance
                    rd_pipe_1 <= rd_pipe_0;

                    // Pipeline stage 1 : issue BRAM read
                    if (bin_idx <= 9'd200) begin
                        coeff_rd_addr <= coeff_base + {5'd0, bin_idx};
                        power_rd_addr <= bin_idx[7:0];
                        bin_idx       <= bin_idx + 9'd1;
                        rd_pipe_0     <= 1'b1;
                    end else begin
                        rd_pipe_0 <= 1'b0;
                        // Wait for pipeline to drain
                        if (!rd_pipe_1 && !rd_pipe_0)
                            me_state <= ME_LOG_WRITE;
                    end
                end

                // -------------------------------------------------
                ME_LOG_WRITE: begin
                    result_mem[mel_idx] <= log_result;

                    if (mel_idx == 7'd79) begin
                        me_state <= ME_DONE;
                    end else begin
                        mel_idx    <= mel_idx + 7'd1;
                        coeff_base <= coeff_base + 14'd201;
                        bin_idx    <= 9'd0;
                        accum      <= 48'd0;
                        rd_pipe_0  <= 1'b0;
                        rd_pipe_1  <= 1'b0;
                        me_state   <= ME_COMPUTE;
                    end
                end

                // -------------------------------------------------
                ME_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    me_state <= ME_IDLE;
                end
            endcase
        end
    end
endmodule
