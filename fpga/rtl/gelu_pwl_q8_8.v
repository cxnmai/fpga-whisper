module gelu_pwl_q8_8 (
    input signed [15:0] x,
    output signed [15:0] y
);
    localparam signed [15:0] SATURATION_POINT = 16'sd768;

    function automatic signed [15:0] gelu_lookup;
        input [3:0] index;
        begin
            case (index)
                4'd0: gelu_lookup = 16'sd0;
                4'd1: gelu_lookup = 16'sd38;
                4'd2: gelu_lookup = 16'sd89;
                4'd3: gelu_lookup = 16'sd148;
                4'd4: gelu_lookup = 16'sd215;
                4'd5: gelu_lookup = 16'sd286;
                4'd6: gelu_lookup = 16'sd358;
                4'd7: gelu_lookup = 16'sd430;
                4'd8: gelu_lookup = 16'sd500;
                4'd9: gelu_lookup = 16'sd569;
                4'd10: gelu_lookup = 16'sd636;
                4'd11: gelu_lookup = 16'sd702;
                default: gelu_lookup = 16'sd767;
            endcase
        end
    endfunction

    function automatic signed [15:0] gelu_nonnegative;
        input signed [15:0] value;
        reg [3:0] index;
        reg [5:0] frac;
        reg signed [15:0] y0;
        reg signed [15:0] y1;
        reg signed [16:0] delta;
        reg signed [31:0] interpolated;
        begin
            if (value >= SATURATION_POINT) begin
                gelu_nonnegative = value;
            end else begin
                index = value[15:6];
                frac = value[5:0];
                y0 = gelu_lookup(index);
                y1 = gelu_lookup(index + 1'b1);
                delta = $signed(y1) - $signed(y0);
                interpolated = $signed(y0) + (($signed(delta) * $signed({1'b0, frac})) >>> 6);
                gelu_nonnegative = interpolated[15:0];
            end
        end
    endfunction

    wire signed [15:0] magnitude = x[15] ? -x : x;
    wire signed [15:0] positive_output = gelu_nonnegative(magnitude);
    assign y = x[15] ? (positive_output - magnitude) : positive_output;
endmodule
