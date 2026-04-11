# program.tcl -- flash the Arty S7-50 over USB-JTAG
#
# Usage:
#   vivado -mode batch -source fpga/scripts/program.tcl
#
# By default this does a volatile RAM load (instant, lost on power-off).
# Set FLASH=1 to write to the onboard SPI flash instead:
#   vivado -mode batch -tclargs -flash -source fpga/scripts/program.tcl

set bit_file  fpga/output/whisper_top.bit
set mcs_file  fpga/output/whisper_top.mcs
set flash_part s25fl128sxxxxxx0-spi-x1_x2_x4

proc set_cfgmem_property_if_supported {cfgmem prop value} {
    if {[lsearch -exact [list_property $cfgmem] $prop] >= 0} {
        set_property $prop $value $cfgmem
    } else {
        puts "Skipping unsupported cfgmem property $prop"
    }
}

# ── parse args ────────────────────────────────────────────────────
set do_flash 0
foreach arg $argv {
    if {$arg eq "-flash"} { set do_flash 1 }
}

# ── connect to board ──────────────────────────────────────────────
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target

set device [lindex [get_hw_devices] 0]
current_hw_device $device

if {$do_flash} {
    # ── persistent: write MCS to SPI flash ────────────────────────
    puts "Writing $mcs_file to SPI flash ($flash_part) ..."

    create_hw_cfgmem -hw_device $device -mem_dev [lindex [get_cfgmem_parts $flash_part] 0]
    set cfgmem [get_property PROGRAM.HW_CFGMEM $device]
    set cfgmem_bitfile [get_property PROGRAM.HW_CFGMEM_BITFILE $device]

    set_cfgmem_property_if_supported $cfgmem PROGRAM.FILES [list $mcs_file]
    set_cfgmem_property_if_supported $cfgmem PROGRAM.UNUSED_PIN_TERMINATION {pull-none}
    set_cfgmem_property_if_supported $cfgmem PROGRAM.BLANK_CHECK {0}
    set_cfgmem_property_if_supported $cfgmem PROGRAM.ERASE {1}
    set_cfgmem_property_if_supported $cfgmem PROGRAM.CFG_PROGRAM {1}
    set_cfgmem_property_if_supported $cfgmem PROGRAM.VERIFY {1}

    # Indirect flash programming first loads a small helper image that
    # bridges the FPGA fabric to the attached SPI flash device.
    create_hw_bitstream -hw_device $device $cfgmem_bitfile
    program_hw_devices $device
    refresh_hw_device $device

    program_hw_cfgmem -hw_cfgmem $cfgmem

    # Boot from flash immediately
    boot_hw_device $device

    puts "Flash programming complete. Design will auto-load on power-up."
} else {
    # ── volatile: load bitstream to FPGA RAM ──────────────────────
    puts "Programming $bit_file to FPGA RAM (volatile) ..."

    set_property PROGRAM.FILE $bit_file $device
    program_hw_devices $device

    puts "FPGA programmed. LED0 should be blinking."
}

close_hw_target
disconnect_hw_server
close_hw_manager
