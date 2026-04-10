module dot_product_i16x8 (
    input signed [15:0] a0,
    input signed [15:0] a1,
    input signed [15:0] a2,
    input signed [15:0] a3,
    input signed [15:0] a4,
    input signed [15:0] a5,
    input signed [15:0] a6,
    input signed [15:0] a7,
    input signed [15:0] b0,
    input signed [15:0] b1,
    input signed [15:0] b2,
    input signed [15:0] b3,
    input signed [15:0] b4,
    input signed [15:0] b5,
    input signed [15:0] b6,
    input signed [15:0] b7,
    output signed [63:0] result
);
    assign result =
        ($signed(a0) * $signed(b0)) +
        ($signed(a1) * $signed(b1)) +
        ($signed(a2) * $signed(b2)) +
        ($signed(a3) * $signed(b3)) +
        ($signed(a4) * $signed(b4)) +
        ($signed(a5) * $signed(b5)) +
        ($signed(a6) * $signed(b6)) +
        ($signed(a7) * $signed(b7));
endmodule
