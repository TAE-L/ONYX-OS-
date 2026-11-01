/* Static user program: text (R-X), rodata (R--) and data+bss (RW-) in separate
   pages so the kernel can map each with exactly the permissions it needs. */
ENTRY(_start)

PHDRS {
    text   PT_LOAD FLAGS(5);
    rodata PT_LOAD FLAGS(4);
    data   PT_LOAD FLAGS(6);
}

SECTIONS {
    . = 0x400000;

    .text : ALIGN(4K) {
        *(.text .text.*)
    } :text

    .rodata : ALIGN(4K) {
        *(.rodata .rodata.*)
    } :rodata

    .data : ALIGN(4K) {
        *(.got .got.*)
        *(.data .data.*)
    } :data

    .bss : ALIGN(4K) {
        *(.bss .bss.*)
        *(COMMON)
    } :data

    /DISCARD/ : {
        *(.eh_frame .eh_frame_hdr .comment .note .note.*)
    }
}
