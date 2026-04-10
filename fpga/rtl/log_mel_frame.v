module log_mel_frame (
    input wire [(201 * 24) - 1:0] power_flat,
    output wire [(80 * 16) - 1:0] log_mel_flat
);
    wire [(80 * 48) - 1:0] mel_flat;

    mel_filterbank_201x80 mel_filterbank (
        .power_flat(power_flat),
        .mel_flat(mel_flat)
    );

    genvar lane_index;
    generate
        for (lane_index = 0; lane_index < 80; lane_index = lane_index + 1) begin : lane_gen
            log_mel_q8_8 log_lane (
                .mel_value(mel_flat[(lane_index * 48) +: 48]),
                .log_value(log_mel_flat[(lane_index * 16) +: 16])
            );
        end
    endgenerate
endmodule
