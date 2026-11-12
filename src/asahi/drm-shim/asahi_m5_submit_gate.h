/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

/*
 * Structural replica of the submission checks the G17 kernel runtime applies
 * to the M5 command set (the M5 Pro and M5 Max, G17S and G17C): the command
 * parser (drivers/gpu/drm/asahi/g17/command.rs, Parser and
 * Command::validate), the program storage lists
 * (g17/render_resources.rs) and their per-entry demand limits
 * (g17/fw/m5/render_demand.rs), the render requirement of the backend
 * (g17/runtime/backend.rs, prepare_command) and the M5 render admission
 * (g17/queue/render/m5_render_submission.rs, validate).
 *
 * It does not model the device's pool budget, the client's address space or
 * the firmware: a submission it accepts can still be refused for a mapping or
 * a pool size, but one it refuses is refused by the kernel too, with the
 * returned errno.
 */

struct asahi_m5_gate_stats {
   unsigned renders, computes;
   unsigned render_programs, compute_programs;
};

/* Returns 0 or the negative errno the kernel returns; *reason names the
 * failed check. */
int asahi_m5_submit_gate(const void *cmdbuf, size_t size, uint64_t features,
                         struct asahi_m5_gate_stats *stats,
                         const char **reason);
