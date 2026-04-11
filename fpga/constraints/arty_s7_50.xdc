## ============================================================================
## Constraints for Digilent Arty S7-50  (XC7S50-CSGA324-1)
## Target design: whisper_top
##
## Reference: Digilent Arty S7-50 master XDC / board schematic Rev E.
## Verify pin assignments against YOUR board revision before programming.
## ============================================================================

## ----------------------------------------------------------------------------
## Clock  --  12 MHz crystal oscillator
## The Arty S7 ships with a 12 MHz oscillator on F14.
## whisper_top uses an internal MMCM to generate 100 MHz from this.
## ----------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN F14  IOSTANDARD LVCMOS33 } [get_ports { clk_12mhz }]
create_clock -name clk_12mhz -period 83.333 -waveform {0.000 41.667} [get_ports { clk_12mhz }]

## ----------------------------------------------------------------------------
## Push buttons  (active-high, active when pressed)
## BTN0 is used as system reset in whisper_top.
## ----------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN G15  IOSTANDARD LVCMOS33 } [get_ports { btn[0] }]
set_property -dict { PACKAGE_PIN K16  IOSTANDARD LVCMOS33 } [get_ports { btn[1] }]
set_property -dict { PACKAGE_PIN J16  IOSTANDARD LVCMOS33 } [get_ports { btn[2] }]
set_property -dict { PACKAGE_PIN H13  IOSTANDARD LVCMOS33 } [get_ports { btn[3] }]

## ----------------------------------------------------------------------------
## Slide switches
## ----------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN H14  IOSTANDARD LVCMOS33 } [get_ports { sw[0] }]
set_property -dict { PACKAGE_PIN H18  IOSTANDARD LVCMOS33 } [get_ports { sw[1] }]
set_property -dict { PACKAGE_PIN G18  IOSTANDARD LVCMOS33 } [get_ports { sw[2] }]
set_property -dict { PACKAGE_PIN M5   IOSTANDARD LVCMOS33 } [get_ports { sw[3] }]

## ----------------------------------------------------------------------------
## Green LEDs  (active-high)
## ----------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN E18  IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN F13  IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN E13  IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN H15  IOSTANDARD LVCMOS33 } [get_ports { led[3] }]

## ----------------------------------------------------------------------------
## RGB LEDs  (accent, accent accent-high accent accent accent-high)
## ----------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN J15  IOSTANDARD LVCMOS33 } [get_ports { led0_r }]
set_property -dict { PACKAGE_PIN G17  IOSTANDARD LVCMOS33 } [get_ports { led0_g }]
set_property -dict { PACKAGE_PIN F15  IOSTANDARD LVCMOS33 } [get_ports { led0_b }]
set_property -dict { PACKAGE_PIN E15  IOSTANDARD LVCMOS33 } [get_ports { led1_r }]
set_property -dict { PACKAGE_PIN F18  IOSTANDARD LVCMOS33 } [get_ports { led1_g }]
set_property -dict { PACKAGE_PIN E14  IOSTANDARD LVCMOS33 } [get_ports { led1_b }]

## ----------------------------------------------------------------------------
## USB-UART  (FTDI FT2232HQ)
##   uart_rxd = host-to-FPGA  (FPGA receives)
##   uart_txd = FPGA-to-host  (FPGA transmits)
## ----------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN V12  IOSTANDARD LVCMOS33 } [get_ports { uart_rxd }]
set_property -dict { PACKAGE_PIN R12  IOSTANDARD LVCMOS33 } [get_ports { uart_txd }]

## UART RX is asynchronous -- mark false path to avoid spurious timing failures.
set_false_path -from [get_ports { uart_rxd }]

## ----------------------------------------------------------------------------
## Pmod Header JA  (active accent accent-high accent)
##   Accent: accent accent accent accent accent accent accent accent.
##   Accent accent accent accent accent for future logic-analyzer / debug.
## ----------------------------------------------------------------------------
#set_property -dict { PACKAGE_PIN L17  IOSTANDARD LVCMOS33 } [get_ports { ja[0] }]
#set_property -dict { PACKAGE_PIN L18  IOSTANDARD LVCMOS33 } [get_ports { ja[1] }]
#set_property -dict { PACKAGE_PIN M14  IOSTANDARD LVCMOS33 } [get_ports { ja[2] }]
#set_property -dict { PACKAGE_PIN N14  IOSTANDARD LVCMOS33 } [get_ports { ja[3] }]
#set_property -dict { PACKAGE_PIN R18  IOSTANDARD LVCMOS33 } [get_ports { ja[4] }]
#set_property -dict { PACKAGE_PIN P18  IOSTANDARD LVCMOS33 } [get_ports { ja[5] }]
#set_property -dict { PACKAGE_PIN M16  IOSTANDARD LVCMOS33 } [get_ports { ja[6] }]
#set_property -dict { PACKAGE_PIN M17  IOSTANDARD LVCMOS33 } [get_ports { ja[7] }]

## ----------------------------------------------------------------------------
## DDR3L SDRAM  (Micron MT41K128M16JT-125, 256 MB)
##   Accent: accent accent accent MIG IP.  The Memory Interface Generator
##   accent accent accent accent accent accent accent accent accent accent.
##   accent accent accent accent accent accent accent accent accent accent.
## ----------------------------------------------------------------------------
## DDR3 pins are managed by Vivado MIG IP -- do not manually constrain them.
## When you add MIG to the design, import its generated XDC instead.

## ----------------------------------------------------------------------------
## Timing
## ----------------------------------------------------------------------------

## The 100 MHz system clock is generated by the MMCM inside whisper_top.
## Vivado auto-derives it; just constrain I/O against the input clock.
set_input_delay  -clock clk_12mhz -max 5.0 [get_ports { btn[*] sw[*] }]
set_input_delay  -clock clk_12mhz -min 0.0 [get_ports { btn[*] sw[*] }]
set_output_delay -clock clk_12mhz -max 5.0 [get_ports { led[*] led0_* led1_* uart_txd }]
set_output_delay -clock clk_12mhz -min 0.0 [get_ports { led[*] led0_* led1_* uart_txd }]

## ----------------------------------------------------------------------------
## Configuration
## ----------------------------------------------------------------------------
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property CONFIG_MODE SPIx4 [current_design]
