# build.tcl -- Vivado non-project-mode synthesis for fpga-whisper
# Target: Digilent Arty S7-50  (XC7S50-CSGA324-1)
#
# Usage:
#   cd <repo-root>
#   vivado -mode batch -source fpga/scripts/build.tcl
#
# Outputs land in fpga/output/.

if {![info exists part]} {
    set part xc7s50csga324-1
}
if {![info exists top]} {
    set top whisper_top
}
if {![info exists output_dir]} {
    set output_dir fpga/output
}

file mkdir $output_dir

# ── read sources ──────────────────────────────────────────────────
set rtl_dir fpga/rtl

read_verilog $rtl_dir/dot_product_i16x8.v
read_verilog $rtl_dir/gemm_tile_i16x8.v
read_verilog $rtl_dir/gemm_tile_accum_i16x8.v
read_verilog $rtl_dir/gelu_pwl_q8_8.v
read_verilog $rtl_dir/gelu_pwl_q8_8x8.v
read_verilog $rtl_dir/log_mel_q8_8.v
read_verilog $rtl_dir/mel_engine_seq.v
read_verilog $rtl_dir/uart_rx.v
read_verilog $rtl_dir/uart_tx.v
read_verilog $rtl_dir/whisper_top.v

# Note: mel_filterbank_201x80.v, log_mel_frame.v, and log_mel_q8_8.v
# are still used for simulation.  For synthesis the sequential
# mel_engine_seq replaces the combinational chain.  log_mel_q8_8 is
# instantiated inside mel_engine_seq.

read_xdc fpga/constraints/arty_s7_50.xdc

# ── synthesis ─────────────────────────────────────────────────────
synth_design -top $top -part $part

write_checkpoint   -force $output_dir/post_synth.dcp
report_timing_summary -file $output_dir/timing_synth.rpt
report_utilization    -file $output_dir/utilization_synth.rpt

# ── implementation ────────────────────────────────────────────────
opt_design
place_design
write_checkpoint   -force $output_dir/post_place.dcp
report_timing_summary -file $output_dir/timing_place.rpt

route_design
write_checkpoint   -force $output_dir/post_route.dcp
report_timing_summary -file $output_dir/timing_route.rpt
report_utilization    -file $output_dir/utilization_route.rpt

# ── bitstream ─────────────────────────────────────────────────────
write_bitstream -force $output_dir/whisper_top.bit

# SPI flash programming image for persistent configuration
write_cfgmem -format mcs -size 16 -interface SPIx4 \
    -loadbit "up 0x0 $output_dir/whisper_top.bit" \
    -force -file $output_dir/whisper_top.mcs

puts "========================================"
puts " Build complete."
puts " Bitstream : $output_dir/whisper_top.bit"
puts " Flash MCS : $output_dir/whisper_top.mcs"
puts "========================================"
