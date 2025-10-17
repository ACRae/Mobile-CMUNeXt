// debug.c
#include "debug.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#if LOG_USE_LIBC_PRINTF
  #include <stdio.h>
  #define LOG_OUT_STR(s)   do { printf("%s", (s)); } while (0)
#else
  #include "xil_printf.h"
  // xil_printf has no vprintf; we preformat to a buffer, then print as %s
  #define LOG_OUT_STR(s)   do { xil_printf("%s", (s)); } while (0)
#endif

// Globals (simple and tiny; change to atomic if you need concurrency)
static volatile int g_log_enabled = 1;
static volatile int g_log_level   = LOG_LEVEL_DEBUG;

static const char* level_tag(log_level_t lvl) {
    switch (lvl) {
        case LOG_LEVEL_ERROR:   return "[ERROR]";
        case LOG_LEVEL_DEBUG:   return "[DEBUG]";
        case LOG_LEVEL_INFO:    return "[INFO]";
        case LOG_LEVEL_VERBOSE: return "[VERBOSE]";
        default:                return "[LOG]";
    }
}

int  log_get_level(void)        { return g_log_level; }
void log_set_level(int level)   { g_log_level = level; }
void log_enable(void)           { g_log_enabled = 1; }
void log_disable(void)          { g_log_enabled = 0; }
int  log_is_enabled(void)       { return g_log_enabled; }

static void log_vprintf_impl(log_level_t level, const char *fmt, va_list ap) {
    if (!g_log_enabled || level > g_log_level) return;

    char msg[LOG_BUFFER_SIZE];
    int n = vsnprintf(msg, sizeof(msg), fmt, ap);
    if (n < 0) return; // formatting failed

    // Ensure NUL-termination even if truncated
    msg[sizeof(msg) - 1] = '\0';

    // Compose final line: "[TAG] " + message + optional '\n'
    char out[LOG_BUFFER_SIZE + 16];
    const char *tag = level_tag(level);

#if LOG_AUTO_NEWLINE
    int m = snprintf(out, sizeof(out), "%s %s\n", tag, msg);
#else
    int m = snprintf(out, sizeof(out), "%s %s",  tag, msg);
#endif
    (void)m; // ignore truncation; it's fine for logs
    LOG_OUT_STR(out);
}

void log_printf(log_level_t level, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_vprintf_impl(level, fmt, ap);
    va_end(ap);
}

void log_error(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_vprintf_impl(LOG_LEVEL_ERROR, fmt, ap);
    va_end(ap);
}
void log_debug(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_vprintf_impl(LOG_LEVEL_DEBUG, fmt, ap);
    va_end(ap);
}
void log_info(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_vprintf_impl(LOG_LEVEL_INFO, fmt, ap);
    va_end(ap);
}
void log_verbose(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_vprintf_impl(LOG_LEVEL_VERBOSE, fmt, ap);
    va_end(ap);
}

void log_hexdump(log_level_t level, const void *data, size_t len) {
    if (!g_log_enabled || level > g_log_level) return;
    const uint8_t *p = (const uint8_t*)data;
    char line[3*16 + 1 + 16 + 1]; // "xx " *16 + space + ascii *16 + NUL

    for (size_t i = 0; i < len; i += 16) {
        size_t chunk = (len - i) < 16 ? (len - i) : 16;
        char *hex = line, *asc = line + 3*16 + 1;

        for (size_t j = 0; j < 16; ++j) {
            if (j < chunk) {
                sprintf(hex, "%02X ", p[i+j]);
                *asc++ = (p[i+j] >= 32 && p[i+j] < 127) ? (char)p[i+j] : '.';
            } else {
                strcpy(hex, "   ");
                *asc++ = ' ';
            }
            hex += 3;
        }
        *hex = ' '; *++hex = '\0';
        *asc = '\0';

        // print one formatted line
        log_printf(level, "%04X: %s %s", (unsigned)i, line, (const char*)(line + 3*16 + 1));
    }
}
