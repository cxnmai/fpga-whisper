module log_mel_q8_8 (
    input wire [47:0] mel_value,
    output wire [15:0] log_value
);
    // Synthesizable log2 in Q8.8 fixed-point.
    // Uses a fixed-iteration priority encoder (no while loop) and
    // replaces division-by-power-of-2 with a right shift.
    function automatic [15:0] log2_linear_q8_8;
        input [47:0] value;
        integer i;
        reg [5:0]  exponent;
        reg [47:0] remainder;
        reg [55:0] numer;
        begin
            if (value == 48'd0) begin
                log2_linear_q8_8 = 16'd0;
            end else begin
                // Priority encoder: find highest set bit position
                exponent = 6'd0;
                for (i = 0; i < 48; i = i + 1)
                    if (value[i]) exponent = i[5:0];

                remainder = value - (48'd1 << exponent);
                // (remainder << 8) / (1 << exponent) == (remainder << 8) >> exponent
                numer = {8'd0, remainder} << 8;
                log2_linear_q8_8 = ({10'd0, exponent} << 8) + (numer >> exponent);
            end
        end
    endfunction

    assign log_value = log2_linear_q8_8(mel_value);
endmodule
