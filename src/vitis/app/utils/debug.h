// debug.h - Lightweight logging for Vitis/Standalone
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Levels (lower = more critical)
typedef enum {
    LOG_LEVEL_ERROR   = 1,
    LOG_LEVEL_DEBUG   = 2,
    LOG_LEVEL_INFO    = 3,
    LOG_LEVEL_VERBOSE = 4
} log_level_t;

// --- Configuration (optional) ---
// Size of the internal formatting buffer used by vsnprintf
#ifndef LOG_BUFFER_SIZE
#define LOG_BUFFER_SIZE 256
#endif
// Append '\n' automatically to each message?
#ifndef LOG_AUTO_NEWLINE
#define LOG_AUTO_NEWLINE 1
#endif
// If you want to route to libc printf instead of xil_printf, define LOG_USE_LIBC_PRINTF 1
#ifndef LOG_USE_LIBC_PRINTF
#define LOG_USE_LIBC_PRINTF 0
#endif

// --- Control ---
void log_set_level(int level);
int  log_get_level(void);
void log_enable(void);
void log_disable(void);
int  log_is_enabled(void);

// --- Logging APIs (printf-style) ---
void log_printf(log_level_t level, const char *fmt, ...);
void log_error (const char *fmt, ...);
void log_debug (const char *fmt, ...);
void log_info  (const char *fmt, ...);
void log_verbose(const char *fmt, ...);

// Optional: quick hexdump helper
void log_hexdump(log_level_t level, const void *data, size_t len);

#ifdef __cplusplus
}
#endif
