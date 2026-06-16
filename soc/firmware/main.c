// SPDX-License-Identifier: MIT
// Reference-SoC firmware (tier2-adoption-checklist §2): a bare-metal RISC-V app
// that drives the motorloop controller over AXI-Lite and streams telemetry over
// the LiteX UART. It writes the control registers to command closed-loop FOC,
// then polls the telemetry registers and prints them. The register map is
// axil_regfile's (rtl/contracts/axil_regfile.md); the peripheral is mapped at
// MOTOR_BASE by soc/motorloop_soc.py.
#include <stdint.h>
#include <stdio.h>
#include <generated/csr.h>
#include <libbase/uart.h>

#define MOTOR_BASE 0xb0000000UL
#define REG(i)     (*(volatile uint32_t *)(MOTOR_BASE + (uint32_t)(i) * 4u))

// Write register indices (byte addr = index*4).
enum { R_MODE = 0, R_DUTY = 1, R_TARGET_SPEED = 2, R_ALIGN = 3,
       R_OL_FREQ_HI = 4, R_OL_FREQ_LO = 5, R_OL_RAMP_HI = 6, R_OL_RAMP_LO = 7,
       R_CONTROL = 8 };
// Read (telemetry) register indices.
enum { T_SPEED = 16, T_FAULTS = 17, T_ANGLE = 18, T_NOCTW = 19,
       T_STATUS = 20, T_FLAGS = 21 };

enum { MODE_IDLE = 0, MODE_OPEN_LOOP = 1, MODE_SIX_STEP = 2, MODE_FOC = 3 };

static void busy_delay(volatile uint32_t n) { while (n--) __asm__ volatile (""); }

int main(void) {
#ifdef CONFIG_CPU_HAS_INTERRUPT
    irq_setmask(0);
    irq_setie(1);
#endif
    uart_init();
    printf("\n[motorloop] reference SoC up; driving the controller over AXI-Lite\n");

    // Command closed-loop FOC at a target speed (torque comes from the speed PI).
    REG(R_MODE)         = MODE_FOC;
    REG(R_TARGET_SPEED) = 0x0200;     // rad/s (placeholder units; see Q1)
    REG(R_CONTROL)      = 1;          // use_axi: take commands from these registers

    printf("[motorloop] mode=%u target_speed=%u\n",
           (unsigned)REG(R_MODE), (unsigned)REG(R_TARGET_SPEED));

    for (;;) {
        uint32_t status = REG(T_STATUS);    // {configured[3], sector[2:0]}
        uint32_t speed  = REG(T_SPEED);
        uint32_t faults = REG(T_FAULTS);    // {fault[15:8], mismatch[7:0]}
        printf("[motorloop] speed=%5u sector=%u configured=%u faults=%u flags=0x%02x\n",
               (unsigned)(speed & 0xffff), (unsigned)(status & 0x7),
               (unsigned)((status >> 3) & 0x1), (unsigned)(faults & 0xff),
               (unsigned)(REG(T_FLAGS) & 0xff));
        busy_delay(2000000);
    }
    return 0;
}
