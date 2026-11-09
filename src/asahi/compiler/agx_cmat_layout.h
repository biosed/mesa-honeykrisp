/*
 * Copyright © 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>

/*
 * One subgroup distributes an 8x8 matrix as two elements per lane.  This is
 * the layout already used by agx_nir_lower_cmat for the native 8x8 matrix
 * instruction.  Keeping the software tile in the same layout lets a future
 * 16x8x16 lowering compose four 8x8x8 operations without repacking the two
 * result tiles between the K=8 steps.
 */
static inline unsigned
agx_cmat_8x8_row(unsigned lane, unsigned element)
{
   assert(lane < 32 && element < 2);
   return ((lane >> 1) & 3) | ((lane >> 2) & 4);
}

static inline unsigned
agx_cmat_8x8_col(unsigned lane, unsigned element)
{
   assert(lane < 32 && element < 2);
   return ((lane & 1) << 1) | ((lane >> 1) & 4) | element;
}

static inline unsigned
agx_cmat_8x8_owner_lane(unsigned row, unsigned col)
{
   assert(row < 8 && col < 8);
   return ((row & 3) << 1) | ((col & 2) >> 1) |
          ((col & 4) << 1) | ((row & 4) << 2);
}

static inline unsigned
agx_cmat_8x8_owner_element(unsigned row, unsigned col)
{
   assert(row < 8 && col < 8);
   return col & 1;
}
