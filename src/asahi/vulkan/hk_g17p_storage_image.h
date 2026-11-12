/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* The table pointers of G17P's state-0 storage-image compute ABI
 * (agx_nir_lower_g17p_storage_image()), loaded into uniform halves 0..15.
 *
 * Write-only image: [0] its PBE descriptor, [1] the write control.
 *
 * Read/write image: [0] its texture/PBE descriptor pair (the read is
 * texture record 0; the store takes PBE record 1, see
 * agx_shader_key::g17p_storage_image_pair), [1] the texel-fetch sampler table
 * (the read's sampler 0 is read at heap base + 4, as for a read-only texel
 * fetch), [2] the paired read/write controls, [3] the coordinate clamp word
 * the native read/write read compares against at uniform half 12.
 *
 * Returns false when the descriptors are not the pair the route requires.
 */
static inline bool
hk_g17p_storage_image_pointers(uint64_t set_va, uint32_t texture_offset_B,
                               uint32_t pbe_offset_B, bool write_only,
                               uint64_t access_control_va,
                               uint64_t txf_sampler_va, uint64_t pointers[4])
{
   if (!set_va || !access_control_va || (access_control_va & 31) ||
       (pbe_offset_B & 7) || set_va > UINT64_MAX - pbe_offset_B - 32)
      return false;

   pointers[2] = pointers[3] = 0;
   if (write_only) {
      pointers[0] = set_va + pbe_offset_B;
      pointers[1] = access_control_va;
      return true;
   }

   if (!txf_sampler_va || pbe_offset_B != texture_offset_B + 32)
      return false;

   pointers[0] = set_va + texture_offset_B;
   pointers[1] = txf_sampler_va;
   pointers[2] = access_control_va;
   pointers[3] = UINT64_C(0xffff);
   return true;
}
