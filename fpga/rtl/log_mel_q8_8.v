module log_mel_q8_8 (
    input wire [47:0] mel_value,
    output wire [15:0] log_value
);
    function automatic [15:0] log2_linear_q8_8;
        input [47:0] value;
        reg [47:0] temp;
        reg [47:0] base;
        reg [47:0] remainder;
        reg [15:0] exponent;
        reg [15:0] fractional;
        begin
            if (value == 48'd0) begin
                log2_linear_q8_8 = 16'd0;
            end else begin
                temp = value;
                exponent = 16'd0;
                while (temp > 48'd1) begin
                    temp = temp >> 1;
                    exponent = exponent + 16'd1;
                end

                base = 48'd1 << exponent;
                remainder = value - base;
                fractional = (remainder << 8) / base;
                log2_linear_q8_8 = (exponent << 8) + fractional;
            end
        end
    endfunction

    assign log_value = log2_linear_q8_8(mel_value);
endmodule
