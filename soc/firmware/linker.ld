/* SPDX-License-Identifier: MIT */
/* Minimal linker script for the motorloop reference-SoC firmware. The concrete
 * region addresses (main_ram / rom) come from the SoC build's generated
 * `regions.ld`, which `INCLUDE`s below - so this stays board/config-agnostic
 * (tier2-adoption-checklist §2). Modelled on the LiteX bare-metal demo. */
INCLUDE generated/output_format.ld
INCLUDE generated/regions.ld
ENTRY(_start)

SECTIONS {
    .text : {
        _ftext = .;
        *(.text.start)
        *(.text .stub .text.* .gnu.linkonce.t.*)
        _etext = .;
    } > main_ram

    .rodata : {
        . = ALIGN(4);
        _frodata = .;
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        *(.rodata1)
        _erodata = .;
    } > main_ram

    .data : {
        . = ALIGN(4);
        _fdata = .;
        *(.data .data.* .gnu.linkonce.d.*)
        *(.data1)
        _edata = .;
    } > main_ram

    .bss : {
        . = ALIGN(4);
        _fbss = .;
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        _ebss = .;
        _end = .;
    } > main_ram
}

PROVIDE(_fstack = ORIGIN(main_ram) + LENGTH(main_ram) - 8);
