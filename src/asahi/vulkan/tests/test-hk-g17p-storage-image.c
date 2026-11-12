/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * The G17P state-0 storage-image pointer table, without a device.
 */
#include <inttypes.h>
#include <stdio.h>
#include "hk_g17p_storage_image.h"

static unsigned tests, failures;

static void
check(const char *name, uint64_t actual, uint64_t expected)
{
   tests++;
   if (actual != expected) {
      fprintf(stderr, "%s: got 0x%" PRIx64 ", expected 0x%" PRIx64 "\n", name,
              actual, expected);
      failures++;
   }
}

int
main(void)
{
   const uint64_t set = 0x5bfd66bf40, control = 0x5bfce3c020;
   const uint64_t sampler = 0x5bfde98000;
   uint64_t p[4];

   /* Read/write: texture/PBE pair, then the texel-fetch sampler the read takes
    * from halves 4..7 (it was the PBE pointer: the read used PBE bytes as its
    * sampler), the controls and the clamp word. */
   check("rw accepted", hk_g17p_storage_image_pointers(set, 0x20, 0x40, false,
                                                       control, sampler, p),
         true);
   check("rw pair", p[0], set + 0x20);
   check("rw sampler", p[1], sampler);
   check("rw controls", p[2], control);
   check("rw clamp", p[3], 0xffff);

   /* Write-only: the PBE, then the write control. */
   check("wo accepted", hk_g17p_storage_image_pointers(set, 0, 0x40, true,
                                                       control, sampler, p),
         true);
   check("wo pbe", p[0], set + 0x40);
   check("wo control", p[1], control);
   check("wo rest", p[2] | p[3], 0);

   /* The route's invariants. */
   check("rw needs the pair", hk_g17p_storage_image_pointers(
            set, 0x20, 0x60, false, control, sampler, p), false);
   check("rw needs a sampler", hk_g17p_storage_image_pointers(
            set, 0x20, 0x40, false, control, 0, p), false);
   check("aligned control", hk_g17p_storage_image_pointers(
            set, 0x20, 0x40, false, control + 8, sampler, p), false);
   check("no set", hk_g17p_storage_image_pointers(0, 0x20, 0x40, true,
                                                  control, sampler, p),
         false);

   printf("%u/%u passed\n", tests - failures, tests);
   return failures ? 1 : 0;
}
