module gelu_pwl_q8_8x8 (
    input signed [8 * 16 - 1:0] input_block,
    output signed [8 * 16 - 1:0] output_block
);
    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin: lane_gen
            gelu_pwl_q8_8 scalar (
                .x(input_block[(lane * 16) +: 16]),
                .y(output_block[(lane * 16) +: 16])
            );
        end
    endgenerate
endmodule
