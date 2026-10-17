/* OnyxOS kernel image: higher half, top 2 GiB (code-model=kernel).
   Every section starts and ends on a 4 KiB boundary because paging.rs maps
   .text / .rodata / .data+.bss separately with different permissions (W^X). */

OUTPUT_FORMAT(elf64-x86-64)
ENTRY(_start)

PHDRS
{
    text   PT_LOAD FLAGS(5); /* R-X */
    rodata PT_LOAD FLAGS(4); /* R-- */
    data   PT_LOAD FLAGS(6); /* RW- */
}

SECTIONS
{
    . = 0xffffffff80000000;

    .text : ALIGN(4K) {
        __text_start = .;
        *(.text .text.*)
        . = ALIGN(4K);
        __text_end = .;
    } :text

    .rodata : ALIGN(4K) {
        __rodata_start = .;
        *(.rodata .rodata.*)
        . = ALIGN(4K);
    } :rodata

    /* Static link: the GOT is fully resolved at link time, so it stays read-only. */
    .got : ALIGN(4K) {
        *(.got .got.*)
        . = ALIGN(4K);
        __rodata_end = .;
    } :rodata

    .data : ALIGN(4K) {
        __data_start = .;
        KEEP(*(.limine_requests))
        *(.data .data.*)
    } :data

    .bss : {
        *(.bss .bss.*)
        *(COMMON)
        /* Boot stack with an unmapped guard page below it. */
        . = ALIGN(4K);
        __boot_stack_guard = .;
        . += 4K;
        __boot_stack_bottom = .;
        . += 256K;
        __boot_stack_top = .;
        __bss_end = .;
    } :data

    /DISCARD/ : {
        *(.eh_frame*)
        *(.note .note.*)
        *(.comment)
    }
}
