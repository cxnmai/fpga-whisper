`timescale 1ns / 1ps

module log_mel_frame_tb;
    localparam integer FFT_BINS = 201;
    localparam integer MEL_BINS = 80;

    reg [23:0] power_mem [0:FFT_BINS - 1];
    reg [(FFT_BINS * 24) - 1:0] power_flat;
    wire [(MEL_BINS * 16) - 1:0] log_mel_flat;

    integer result_file;
    integer index;

    log_mel_frame dut (
        .power_flat(power_flat),
        .log_mel_flat(log_mel_flat)
    );

    function automatic [15:0] lane_at;
        input integer lane_index;
        begin
            lane_at = log_mel_flat[(lane_index * 16) +: 16];
        end
    endfunction

    initial begin
        $readmemh("power_spectrum.mem", power_mem);
        for (index = 0; index < FFT_BINS; index = index + 1) begin
            power_flat[(index * 24) +: 24] = power_mem[index];
        end

        result_file = $fopen("logmel_result.txt", "w");
        if (result_file == 0) begin
            $display("failed to open logmel_result.txt");
            $finish;
        end

        #1;
        for (index = 0; index < MEL_BINS; index = index + 1) begin
            $fdisplay(result_file, "%0d", lane_at(index));
        end
        $fclose(result_file);
        $finish;
    end

    initial begin
        $dumpfile("log_mel_frame_tb.vcd");
        $dumpvars(0, log_mel_frame_tb);
    end
endmodule
