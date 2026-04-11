// whisper_top -- top-level for FPGA-Whisper on Digilent Arty S7-50
//
// Host communicates over 115 200-baud UART with a simple binary protocol:
//
//   Host -> FPGA:  [0xAA] [CMD] [LEN_HI] [LEN_LO] [PAYLOAD ...]
//   FPGA -> Host:  [0xAA] [STATUS] [LEN_HI] [LEN_LO] [DATA ...]
//
// Commands
//   0x01  PING            payload 0 B      -> 4 B  (version)
//   0x02  DOT_PRODUCT     payload 32 B     -> 8 B  (i64 result, LE)
//   0x03  GELU_BLOCK      payload 16 B     -> 16 B (8 x i16, LE)
//   0x04  LOAD_MEL_COEFF  payload 32 160 B -> 1 B  (ACK)
//   0x05  MEL_FRAME       payload 603 B    -> 160 B (80 x i16, LE)
//
// Status bytes: 0x00 = OK, 0x01 = ERROR.

module whisper_top #(
    parameter integer CLK_FREQ  = 100_000_000,
    parameter integer BAUD_RATE = 115_200
) (
    input  wire       clk_12mhz,
    input  wire [3:0] btn,
    input  wire [3:0] sw,
    input  wire       uart_rxd,
    output wire       uart_txd,
    output wire [3:0] led,
    output wire       led0_r,
    output wire       led0_g,
    output wire       led0_b,
    output wire       led1_r,
    output wire       led1_g,
    output wire       led1_b
);

    // ================================================================
    //  Constants
    // ================================================================
    localparam [7:0]
        SYNC_BYTE     = 8'hAA,
        STATUS_OK     = 8'h00,
        STATUS_ERROR  = 8'h01,
        CMD_PING           = 8'h01,
        CMD_DOT_PRODUCT    = 8'h02,
        CMD_GELU_BLOCK     = 8'h03,
        CMD_LOAD_MEL_COEFF = 8'h04,
        CMD_MEL_FRAME      = 8'h05;

    localparam [7:0]
        FW_MAJOR = 8'h01,
        FW_MINOR = 8'h00,
        FW_PATCH = 8'h00;

    // ================================================================
    //  MMCM: 12 MHz -> 100 MHz  (synthesis only; sim uses clk_12mhz)
    //    VCO = 12 MHz * 50 = 600 MHz
    //    CLKOUT0 = 600 / 6 = 100 MHz
    // ================================================================
`ifdef SYNTHESIS
    wire clk_mmcm;
    wire mmcm_locked;
    wire mmcm_fb;

    MMCME2_BASE #(
        .CLKIN1_PERIOD   (83.333),   // 12 MHz input
        .CLKFBOUT_MULT_F (50.000),   // VCO = 600 MHz
        .DIVCLK_DIVIDE   (1),
        .CLKOUT0_DIVIDE_F(6.000),    // 100 MHz output
        .STARTUP_WAIT    ("FALSE")
    ) u_mmcm (
        .CLKIN1   (clk_12mhz),
        .CLKFBIN  (mmcm_fb),
        .CLKFBOUT (mmcm_fb),
        .CLKOUT0  (clk_mmcm),
        .LOCKED   (mmcm_locked),
        .PWRDWN   (1'b0),
        .RST      (1'b0)
    );

    BUFG u_bufg (.I(clk_mmcm), .O(clk));
    wire clk;
`else
    // Simulation: use input clock directly
    wire clk = clk_12mhz;
    wire mmcm_locked = 1'b1;
`endif

    // ================================================================
    //  Reset synchroniser  (BTN0 active-high, held until MMCM locks)
    // ================================================================
    reg rst_meta, rst_sync;
    always @(posedge clk) begin
        rst_meta <= btn[0] | ~mmcm_locked;
        rst_sync <= rst_meta;
    end
    wire rst = rst_sync;

    // ================================================================
    //  UART
    // ================================================================
    wire [7:0] rx_data;
    wire       rx_valid;
    reg  [7:0] tx_byte;
    reg        tx_go;
    wire       tx_busy;

    uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rxd),
        .rx_data(rx_data), .rx_valid(rx_valid)
    );
    uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE)) u_tx (
        .clk(clk), .rst(rst),
        .tx_data(tx_byte), .tx_start(tx_go),
        .tx(uart_txd), .tx_busy(tx_busy)
    );

    // ================================================================
    //  Small-payload buffer  (dot product 32 B, gelu 16 B)
    // ================================================================
    reg [7:0] pbuf [0:31];

    // ================================================================
    //  Response buffer  (up to 16 bytes for small commands)
    // ================================================================
    reg [7:0] resp_buf [0:15];

    // ================================================================
    //  Dot product  (combinational, driven from pbuf)
    // ================================================================
    wire [15:0] dp_a0 = {pbuf[1],  pbuf[0]};
    wire [15:0] dp_a1 = {pbuf[3],  pbuf[2]};
    wire [15:0] dp_a2 = {pbuf[5],  pbuf[4]};
    wire [15:0] dp_a3 = {pbuf[7],  pbuf[6]};
    wire [15:0] dp_a4 = {pbuf[9],  pbuf[8]};
    wire [15:0] dp_a5 = {pbuf[11], pbuf[10]};
    wire [15:0] dp_a6 = {pbuf[13], pbuf[12]};
    wire [15:0] dp_a7 = {pbuf[15], pbuf[14]};
    wire [15:0] dp_b0 = {pbuf[17], pbuf[16]};
    wire [15:0] dp_b1 = {pbuf[19], pbuf[18]};
    wire [15:0] dp_b2 = {pbuf[21], pbuf[20]};
    wire [15:0] dp_b3 = {pbuf[23], pbuf[22]};
    wire [15:0] dp_b4 = {pbuf[25], pbuf[24]};
    wire [15:0] dp_b5 = {pbuf[27], pbuf[26]};
    wire [15:0] dp_b6 = {pbuf[29], pbuf[28]};
    wire [15:0] dp_b7 = {pbuf[31], pbuf[30]};

    wire signed [63:0] dp_result;
    dot_product_i16x8 u_dot (
        .a0(dp_a0), .a1(dp_a1), .a2(dp_a2), .a3(dp_a3),
        .a4(dp_a4), .a5(dp_a5), .a6(dp_a6), .a7(dp_a7),
        .b0(dp_b0), .b1(dp_b1), .b2(dp_b2), .b3(dp_b3),
        .b4(dp_b4), .b5(dp_b5), .b6(dp_b6), .b7(dp_b7),
        .result(dp_result)
    );

    // ================================================================
    //  GELU 8-lane  (combinational, driven from pbuf)
    // ================================================================
    wire [127:0] gelu_input = {
        pbuf[15], pbuf[14], pbuf[13], pbuf[12],
        pbuf[11], pbuf[10], pbuf[9],  pbuf[8],
        pbuf[7],  pbuf[6],  pbuf[5],  pbuf[4],
        pbuf[3],  pbuf[2],  pbuf[1],  pbuf[0]
    };
    wire [127:0] gelu_output;
    gelu_pwl_q8_8x8 u_gelu (
        .input_block  (gelu_input),
        .output_block (gelu_output)
    );

    // ================================================================
    //  Mel engine  (sequential, with internal BRAMs)
    // ================================================================
    reg        mel_coeff_wr_en;
    reg [13:0] mel_coeff_wr_addr;
    reg [15:0] mel_coeff_wr_data;
    reg        mel_power_wr_en;
    reg [7:0]  mel_power_wr_addr;
    reg [23:0] mel_power_wr_data;
    reg        mel_start;
    wire       mel_done;
    wire       mel_busy;
    reg  [6:0] mel_result_addr;
    wire [15:0] mel_result_data;

    mel_engine_seq u_mel (
        .clk            (clk),
        .rst            (rst),
        .coeff_wr_en    (mel_coeff_wr_en),
        .coeff_wr_addr  (mel_coeff_wr_addr),
        .coeff_wr_data  (mel_coeff_wr_data),
        .power_wr_en    (mel_power_wr_en),
        .power_wr_addr  (mel_power_wr_addr),
        .power_wr_data  (mel_power_wr_data),
        .start          (mel_start),
        .done           (mel_done),
        .busy           (mel_busy),
        .result_rd_addr (mel_result_addr),
        .result_rd_data (mel_result_data)
    );

    // ================================================================
    //  Command FSM
    // ================================================================
    localparam [3:0]
        S_IDLE      = 4'd0,
        S_CMD       = 4'd1,
        S_LEN_HI    = 4'd2,
        S_LEN_LO    = 4'd3,
        S_PAYLOAD   = 4'd4,
        S_EXEC      = 4'd5,
        S_WAIT_MEL  = 4'd6,
        S_RESPOND   = 4'd7;

    reg [3:0]  state;
    reg [7:0]  cmd_reg;
    reg [15:0] payload_len;
    reg [15:0] payload_cnt;

    // Response control
    reg [7:0]  resp_status;
    reg [15:0] resp_len;      // data bytes (excluding 4-byte header)
    reg [15:0] resp_phase;    // 0=sync, 1=status, 2=len_hi, 3=len_lo, 4..=data

    // Byte-assembly helpers for multi-byte payloads
    reg [7:0]  coeff_lo;           // low byte of 16-bit coefficient
    reg [7:0]  power_b0, power_b1; // first two bytes of 24-bit power value
    reg [1:0]  power_phase;        // 0,1,2 for 3-byte assembly
    reg [7:0]  power_wr_cnt;       // word counter for power BRAM writes

    // Error flag (sticky until next command)
    reg error_flag;

    // Mel result byte mux (combinational, driven from resp_phase)
    wire [15:0] resp_data_idx  = resp_phase - 16'd4;
    wire [6:0]  mel_word_addr  = resp_data_idx[7:1];
    wire        mel_byte_sel   = resp_data_idx[0];

    // Keep the BRAM address always driven so data is ready well before TX
    always @(posedge clk) begin
        if (state == S_RESPOND && cmd_reg == CMD_MEL_FRAME)
            mel_result_addr <= mel_word_addr;
        else
            mel_result_addr <= 7'd0;
    end

    wire [7:0] mel_resp_byte = mel_byte_sel ? mel_result_data[15:8]
                                            : mel_result_data[7:0];

    // ================================================================
    //  Main FSM
    // ================================================================
    integer gi;  // loop variable for GELU serialisation

    always @(posedge clk) begin
        if (rst) begin
            state            <= S_IDLE;
            tx_go            <= 1'b0;
            mel_coeff_wr_en  <= 1'b0;
            mel_power_wr_en  <= 1'b0;
            mel_start        <= 1'b0;
            error_flag       <= 1'b0;
            cmd_reg          <= 8'd0;
            payload_len      <= 16'd0;
            payload_cnt      <= 16'd0;
            resp_status      <= STATUS_OK;
            resp_len         <= 16'd0;
            resp_phase       <= 16'd0;
        end else begin
            // Defaults: one-shot signals clear each cycle
            mel_coeff_wr_en <= 1'b0;
            mel_power_wr_en <= 1'b0;
            mel_start       <= 1'b0;

            case (state)

            // --------------------------------------------------------
            // IDLE -- wait for sync byte 0xAA
            // --------------------------------------------------------
            S_IDLE: begin
                tx_go <= 1'b0;
                if (rx_valid && rx_data == SYNC_BYTE)
                    state <= S_CMD;
            end

            // --------------------------------------------------------
            // CMD -- capture command byte
            // --------------------------------------------------------
            S_CMD: begin
                if (rx_valid) begin
                    cmd_reg <= rx_data;
                    state   <= S_LEN_HI;
                end
            end

            // --------------------------------------------------------
            // LEN -- two bytes, big-endian
            // --------------------------------------------------------
            S_LEN_HI: begin
                if (rx_valid) begin
                    payload_len[15:8] <= rx_data;
                    state <= S_LEN_LO;
                end
            end

            S_LEN_LO: begin
                if (rx_valid) begin
                    payload_len[7:0] <= rx_data;
                    payload_cnt      <= 16'd0;
                    power_phase      <= 2'd0;
                    power_wr_cnt     <= 8'd0;

                    if ({payload_len[15:8], rx_data} == 16'd0)
                        state <= S_EXEC;   // no payload
                    else
                        state <= S_PAYLOAD;
                end
            end

            // --------------------------------------------------------
            // PAYLOAD -- receive data bytes, route by command
            // --------------------------------------------------------
            S_PAYLOAD: begin
                if (rx_valid) begin
                    case (cmd_reg)

                    // -- Coefficient loading: 2 bytes -> 1 x u16 ----
                    CMD_LOAD_MEL_COEFF: begin
                        if (payload_cnt[0] == 1'b0) begin
                            coeff_lo <= rx_data;
                        end else begin
                            mel_coeff_wr_en   <= 1'b1;
                            mel_coeff_wr_addr <= payload_cnt[14:1];
                            mel_coeff_wr_data <= {rx_data, coeff_lo};
                        end
                    end

                    // -- Power spectrum: 3 bytes -> 1 x u24 ---------
                    CMD_MEL_FRAME: begin
                        case (power_phase)
                            2'd0: begin
                                power_b0    <= rx_data;
                                power_phase <= 2'd1;
                            end
                            2'd1: begin
                                power_b1    <= rx_data;
                                power_phase <= 2'd2;
                            end
                            2'd2: begin
                                mel_power_wr_en   <= 1'b1;
                                mel_power_wr_addr <= power_wr_cnt;
                                mel_power_wr_data <= {rx_data, power_b1, power_b0};
                                power_wr_cnt      <= power_wr_cnt + 8'd1;
                                power_phase       <= 2'd0;
                            end
                            default: power_phase <= 2'd0;
                        endcase
                    end

                    // -- Small commands: buffer raw bytes ------------
                    default: begin
                        if (payload_cnt < 16'd32)
                            pbuf[payload_cnt[4:0]] <= rx_data;
                    end

                    endcase

                    payload_cnt <= payload_cnt + 16'd1;
                    if (payload_cnt + 16'd1 == payload_len)
                        state <= S_EXEC;
                end
            end

            // --------------------------------------------------------
            // EXEC -- set up the response for the received command
            // --------------------------------------------------------
            S_EXEC: begin
                error_flag  <= 1'b0;
                resp_phase  <= 16'd0;

                case (cmd_reg)
                CMD_PING: begin
                    resp_status <= STATUS_OK;
                    resp_len    <= 16'd4;
                    resp_buf[0] <= FW_MAJOR;
                    resp_buf[1] <= FW_MINOR;
                    resp_buf[2] <= FW_PATCH;
                    resp_buf[3] <= 8'h00;
                    state       <= S_RESPOND;
                end

                CMD_DOT_PRODUCT: begin
                    resp_status  <= STATUS_OK;
                    resp_len     <= 16'd8;
                    resp_buf[0]  <= dp_result[7:0];
                    resp_buf[1]  <= dp_result[15:8];
                    resp_buf[2]  <= dp_result[23:16];
                    resp_buf[3]  <= dp_result[31:24];
                    resp_buf[4]  <= dp_result[39:32];
                    resp_buf[5]  <= dp_result[47:40];
                    resp_buf[6]  <= dp_result[55:48];
                    resp_buf[7]  <= dp_result[63:56];
                    state        <= S_RESPOND;
                end

                CMD_GELU_BLOCK: begin
                    resp_status <= STATUS_OK;
                    resp_len    <= 16'd16;
                    for (gi = 0; gi < 16; gi = gi + 1)
                        resp_buf[gi] <= gelu_output[gi*8 +: 8];
                    state <= S_RESPOND;
                end

                CMD_LOAD_MEL_COEFF: begin
                    resp_status <= STATUS_OK;
                    resp_len    <= 16'd1;
                    resp_buf[0] <= 8'h01;
                    state       <= S_RESPOND;
                end

                CMD_MEL_FRAME: begin
                    mel_start <= 1'b1;
                    state     <= S_WAIT_MEL;
                end

                default: begin
                    resp_status <= STATUS_ERROR;
                    resp_len    <= 16'd1;
                    resp_buf[0] <= 8'hFF;  // unknown command
                    error_flag  <= 1'b1;
                    state       <= S_RESPOND;
                end
                endcase
            end

            // --------------------------------------------------------
            // WAIT_MEL -- mel engine is computing
            // --------------------------------------------------------
            S_WAIT_MEL: begin
                mel_start <= 1'b0;
                if (mel_done) begin
                    resp_status <= STATUS_OK;
                    resp_len    <= 16'd160;
                    resp_phase  <= 16'd0;
                    state       <= S_RESPOND;
                end
            end

            // --------------------------------------------------------
            // RESPOND -- send header + data over UART
            // --------------------------------------------------------
            S_RESPOND: begin
                if (tx_go) begin
                    // One-shot: clear tx_go the cycle after it was asserted
                    tx_go <= 1'b0;
                end else if (!tx_busy) begin
                    // Compose the next byte
                    case (resp_phase)
                        16'd0: tx_byte <= SYNC_BYTE;
                        16'd1: tx_byte <= resp_status;
                        16'd2: tx_byte <= resp_len[15:8];
                        16'd3: tx_byte <= resp_len[7:0];
                        default: begin
                            if (cmd_reg == CMD_MEL_FRAME)
                                tx_byte <= mel_resp_byte;
                            else
                                tx_byte <= resp_buf[resp_data_idx[3:0]];
                        end
                    endcase

                    tx_go <= 1'b1;

                    if (resp_phase == resp_len + 16'd3)
                        state <= S_IDLE;
                    else
                        resp_phase <= resp_phase + 16'd1;
                end
            end

            default: state <= S_IDLE;

            endcase
        end
    end

    // ================================================================
    //  Status LEDs
    // ================================================================

    // LED 0 : heartbeat  (~1.5 Hz blink proves the clock is alive)
    reg [25:0] hb_cnt;
    always @(posedge clk) begin
        if (rst) hb_cnt <= 26'd0;
        else     hb_cnt <= hb_cnt + 26'd1;
    end
    assign led[0] = hb_cnt[25];

    // LED 1 : toggles on every received UART byte
    reg led1_toggle;
    always @(posedge clk) begin
        if (rst)      led1_toggle <= 1'b0;
        else if (rx_valid) led1_toggle <= ~led1_toggle;
    end
    assign led[1] = led1_toggle;

    // LED 2 : mel engine busy
    assign led[2] = mel_busy;

    // LED 3 : sticky error
    assign led[3] = error_flag;

    // RGB LED 0: FSM state (off / green=idle / blue=rx / red=error)
    assign led0_r = error_flag;
    assign led0_g = (state == S_IDLE);
    assign led0_b = (state == S_PAYLOAD || state == S_RESPOND);

    // RGB LED 1: mel engine (green=done pulse, blue=busy)
    assign led1_r = 1'b0;
    assign led1_g = mel_done;
    assign led1_b = mel_busy;

endmodule
