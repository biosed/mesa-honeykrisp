/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded HAL300 PPP depth/stencil state-block representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_HAL300_DEPTH_STENCIL_CORE_LENGTH 24u
#define AGX_PPP_HAL300_DEPTH_STENCIL_HEADER UINT32_C(0x00000001)

struct agx_ppp_hal300_depth_stencil_fields {
   uint32_t fragment_control_raw;
   uint32_t fragment_front_face_raw;
   uint32_t fragment_back_face_raw;
   uint32_t fragment_front_stencil_raw;
   uint32_t fragment_back_stencil_raw;
};

struct agx_ppp_hal300_depth_stencil_core {
   uint8_t bytes[AGX_PPP_HAL300_DEPTH_STENCIL_CORE_LENGTH];
   uint8_t length;
};

/*
 * Pack/unpack only the exact six-word HAL300 PPP block recovered from the
 * inspected G17G fixed-layout render-state path:
 *
 *   u32 header = bit 0
 *   u32 fragment control
 *   u32 fragment front face
 *   u32 fragment back face
 *   u32 fragment front stencil
 *   u32 fragment back stencil
 *
 * On G17G the default block is 24 bytes: header bit 0 followed by five
 * depth/stencil payload words.  Dynamic state uses the same five slots in
 * the order above.  On G16 the front stencil word comes before the back
 * face word instead, so the adjacent-word swap is HAL300-only.
 *
 * The admitted chip identities are the three HAL300 G17 GPU classes covered by
 * the G17G/G17X data, and G18G, whose J873 native renders carry this block
 * (00000001 000007aa e2e00000 f0c00000 00023020 0001ffff) in every mode.  This
 * codec neither emits the surrounding VDM PPP pointer nor admits a device or
 * advertises a feature.
 */
bool agx_ppp_hal300_depth_stencil_core_supported(enum agx_chip chip);

bool agx_ppp_hal300_depth_stencil_core_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal300_depth_stencil_fields *fields,
   struct agx_ppp_hal300_depth_stencil_core *out);

bool agx_ppp_hal300_depth_stencil_core_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_ppp_hal300_depth_stencil_fields *fields);

#ifdef __cplusplus
}
#endif
