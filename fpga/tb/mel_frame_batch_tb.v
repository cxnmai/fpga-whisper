`timescale 1ns / 1ps

module mel_frame_batch_tb;
    parameter integer FRAME_COUNT = 1;
    localparam integer FFT_BINS = 201;
    localparam integer MEL_BINS = 80;

    reg [23:0] power_mem [0:(FRAME_COUNT * FFT_BINS) - 1];
    reg [(FFT_BINS * 24) - 1:0] power_flat;
    wire [(MEL_BINS * 48) - 1:0] mel_flat;

    integer result_file;
    integer frame_index;
    integer bin_index;
    integer mel_index;

    mel_filterbank_201x80 dut (
        .power_flat(power_flat),
        .mel_flat(mel_flat)
    );

    function automatic [47:0] lane_at;
        input integer lane_index;
        begin
            lane_at = mel_flat[(lane_index * 48) +: 48];
        end
    endfunction

    initial begin
        $readmemh("power_frames.mem", power_mem);

        result_file = $fopen("mel_frame_batch_result.txt", "w");
        if (result_file == 0) begin
            $display("failed to open mel_frame_batch_result.txt");
            $finish;
        end

        for (frame_index = 0; frame_index < FRAME_COUNT; frame_index = frame_index + 1) begin
            for (bin_index = 0; bin_index < FFT_BINS; bin_index = bin_index + 1) begin
                power_flat[(bin_index * 24) +: 24] = power_mem[(frame_index * FFT_BINS) + bin_index];
            end
            #1;
            for (mel_index = 0; mel_index < MEL_BINS; mel_index = mel_index + 1) begin
                $fdisplay(result_file, "%0d", lane_at(mel_index));
            end
        end

        $fclose(result_file);
        $finish;
    end

    initial begin
        $dumpfile("mel_frame_batch_tb.vcd");
        $dumpvars(0, mel_frame_batch_tb);
    end
endmodule
