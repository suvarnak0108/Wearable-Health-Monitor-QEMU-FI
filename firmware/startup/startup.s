.syntax unified
.cpu cortex-m4
.fpu fpv4-sp-d16
.thumb

.global _start
.global Reset_Handler

.section .isr_vector, "a", %progbits
.word _estack
.word Reset_Handler

.text
.thumb_func
Reset_Handler:
    /* Set stack pointer explicitly for QEMU -kernel ELF boot */
    ldr sp, =_estack

    /* Enable CP10 & CP11 Full Access (FPU) */
    ldr r0, =0xE000ED88
    ldr r1, [r0]
    orr r1, r1, #(0xF << 20)
    str r1, [r0]
    dsb
    isb

    /* Paint top 8 KB below _estack with 0xDEADBEEF for runtime stack high-water mark */
    ldr r1, =_estack
    sub r0, r1, #8192
    ldr r2, =0xDEADBEEF
paint_loop:
    cmp r0, r1
    bge paint_done
    str r2, [r0], #4
    b paint_loop
paint_done:
    bl main
loop_forever:
    b loop_forever
