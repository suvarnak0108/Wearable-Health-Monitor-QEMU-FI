static void semihost_write0(const char *message)
{
    register int operation __asm__("r0") = 0x04;
    register const char *argument __asm__("r1") = message;

    __asm__ volatile (
        "bkpt 0xAB"
        : "+r"(operation)
        : "r"(argument)
        : "memory"
    );
}

int main(void)
{
    semihost_write0("Cortex-M4 firmware running in QEMU!\n");

    while (1)
    {
        __asm__ volatile ("wfi");
    }
}
