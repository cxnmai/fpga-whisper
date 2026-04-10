module mel_filterbank_201x80 (
    input wire [(201 * 24) - 1:0] power_flat,
    output reg [(80 * 48) - 1:0] mel_flat
);
    localparam integer FFT_BINS = 201;
    localparam integer MEL_BINS = 80;

    reg [15:0] coeff_mem [0:(FFT_BINS * MEL_BINS) - 1];

    integer mel_index;
    integer bin_index;
    integer coeff_index;
    reg [47:0] accumulator;
    reg [23:0] power_value;
    reg [15:0] coeff_value;

    initial begin
        $readmemh("mel_coeff.mem", coeff_mem);
    end

    always @* begin
        for (mel_index = 0; mel_index < MEL_BINS; mel_index = mel_index + 1) begin
            accumulator = 48'd0;
            for (bin_index = 0; bin_index < FFT_BINS; bin_index = bin_index + 1) begin
                coeff_index = (mel_index * FFT_BINS) + bin_index;
                power_value = power_flat[(bin_index * 24) +: 24];
                coeff_value = coeff_mem[coeff_index];
                accumulator = accumulator + (power_value * coeff_value);
            end
            mel_flat[(mel_index * 48) +: 48] = accumulator;
        end
    end
endmodule
