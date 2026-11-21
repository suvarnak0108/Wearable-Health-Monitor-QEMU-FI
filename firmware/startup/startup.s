.syntax unified
.cpu cortex-m4
.thumb

.global Reset_Handler
.global Default_Handler
.global __Vectors

.extern main
.extern _estack
.extern _sidata
.extern _sdata
.extern _edata
.extern _sbss
.extern _ebss

.section .isr_vector, "a", %progbits
.type __Vectors, %object

__Vectors:
    .word _estack
    .word Reset_Handler
    .rept 46
    .word Default_Handler
    .endr

.section .text.Reset_Handler
.type Reset_Handler, %function

Reset_Handler:
    /* Enable Cortex-M4 Floating Point Unit */
    ldr r0, =0xE000ED88
    ldr r1, =0x00F00000
    ldr r2, [r0]
    orrs r2, r2, r1
    str r2, [r0]

    dsb
    isb
    /* Copy initialized data from flash to RAM */
    ldr r0, =_sidata
    ldr r1, =_sdata
    ldr r2, =_edata

copy_data:
    cmp r1, r2
    bge zero_bss
    ldr r3, [r0], #4
    str r3, [r1], #4
    b copy_data

zero_bss:
    ldr r1, =_sbss
    ldr r2, =_ebss
    movs r3, #0

zero_loop:
    cmp r1, r2
    bge call_main
    str r3, [r1], #4
    b zero_loop

call_main:
    bl main

stop:
    b stop

.size Reset_Handler, .-Reset_Handler

.section .text.Default_Handler
Default_Handler:
    b Default_Handler
