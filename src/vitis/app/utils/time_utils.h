#include "xiltimer.h"
#include <xil_printf.h>
#include <stdio.h>

#define TIME_FUNCTION(description, func_call) do { \
    XTime t0, t1; \
    log_info("Starting: %s\n", description); \
    XTime_GetTime(&t0); \
    func_call; \
    XTime_GetTime(&t1); \
    XTime cycles = t1 - t0; \
    uint32_t elapsed_us_total = (uint32_t)((cycles * 1000000ULL) / COUNTS_PER_SECOND); \
    uint32_t us_whole = elapsed_us_total; \
    uint32_t us_frac = (uint32_t)(((cycles * 10000000ULL) / COUNTS_PER_SECOND) % 10); \
    xil_printf("Completed: %s - Cycles: %lu, Time: %lu.%lu us\r\n", description, (unsigned long)cycles, (unsigned long)us_whole, (unsigned long)us_frac); \
} while(0)
