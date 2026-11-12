/* SPDX-License-Identifier: MIT */
#ifndef AGX_RT_IFT_H
#define AGX_RT_IFT_H
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* HAL300 Gen2 intersection function entry, 16 bytes, eight entries per block.
 * Fields 3/4/6 refer to the native signature record; their full semantic names
 * are not established. Keep them explicit instead of inferring them from the
 * caller's register count. This helper does not compile a traversal program.
 */
static inline bool
agx_rt_ift_entry_pack(uint8_t out[16], uint64_t address,
                      uint32_t resources0, uint32_t resources1,
                      unsigned field3, unsigned field4, unsigned field6,
                      bool kind2)
{
   if (address == 0 || (address & 63) || (address >> 48) ||
       resources0 >= (1u << 18) || resources1 >= (1u << 12) ||
       field3 > 7 || field4 > 7 || field6 > 255)
      return false;
   uint64_t lo = address | ((uint64_t)(resources0 & 0xffff) << 48);
   uint64_t hi = (resources0 >> 16) | ((uint64_t)resources1 << 2) |
                 ((uint64_t)field3 << 32) | ((uint64_t)field4 << 35) |
                 ((uint64_t)(field4 != 0) << 38) | ((uint64_t)field6 << 39) |
                 (kind2 ? UINT64_C(0x8000030000000000) :
                          UINT64_C(0x8000010000000000));
   for (unsigned i = 0; i < 8; ++i) {
      out[i] = lo >> (8 * i);
      out[8 + i] = hi >> (8 * i);
   }
   return true;
}

/* G17P HAL200 Gen1 ordinary user intersection entry. Bit 39 is always
 * set; it has neither HAL300's variable field6 nor its kind2 mode.
 * This does not construct the built-in driver table or its RIA2 programs.
 */
static inline bool
agx_rt_ift_gen1_entry_pack(uint8_t out[16], uint64_t address,
                         uint32_t resources0, uint32_t resources1,
                         unsigned field3, unsigned field4)
{
   return agx_rt_ift_entry_pack(out, address, resources0, resources1,
                               field3, field4, 1, false);
}

/* The native Gen3 table constructor emits 211 program signatures into 112
 * blocks. Seven extra slots alias the default program. These are ABI selectors,
 * not Vulkan shader stages; retain numeric names until their semantics have
 * independent witnesses. */
#define AGX_RT_IFT_GEN3_PROGRAMS 211u
#define AGX_RT_IFT_GEN3_BLOCKS 112u
#define AGX_RT_IFT_GEN3_BYTES (AGX_RT_IFT_GEN3_BLOCKS * 128u)

struct agx_rt_ift_signature {
   uint8_t type, slot, variant, field3, field4, field5, field6;
   uint8_t additional_slots;
};

static inline bool
agx_rt_ift_gen3_signature(unsigned index, struct agx_rt_ift_signature *out)
{
   if (!out || index >= AGX_RT_IFT_GEN3_PROGRAMS)
      return false;
   struct agx_rt_ift_signature s = {0};
   if (index == 0) {
      s.field4 = 4;
      s.field5 = 1;
      s.additional_slots = 7;
   } else if (index <= 128) {
      unsigned i = index - 1, within = i % 16;
      s.variant = i / 16;
      s.field6 = 1;
      if (within < 12) {
         s.type = (within & 1) ? 1 : 2;
         s.slot = within / 2;
         s.field4 = (s.type == 2) ? 2 : 0;
      } else {
         s.type = (within < 14) ? 5 : 3;
         s.slot = within & 1;
         s.field4 = 2;
      }
   } else if (index <= 208) {
      unsigned i = index - 129, within = i % 10;
      s.variant = i / 10;
      if (within < 2) {
         s.type = 9;
         s.slot = within;
      } else if (within < 6) {
         s.type = 10;
         s.slot = within - 2;
      } else {
         s.type = 11;
         s.slot = within - 6;
      }
      s.field6 = s.type == 11 || !(s.slot & 1);
   } else {
      s.type = 4;
      s.slot = index - 209;
      s.field4 = 2;
   }
   *out = s;
   return true;
}

static inline unsigned
agx_rt_ift_gen3_block(const struct agx_rt_ift_signature *s)
{
   static const uint8_t base[13] = {
      104, 96, 0, 8, 56, 64, 0, 0, 0, 72, 80, 88, 112,
   };
   return base[s->type] + s->variant;
}

struct agx_rt_ift_program {
   /* Address of the complete native entry program, including its loader. */
   uint64_t address;
   uint32_t resources0, resources1;
};

/* Construct a whole table transactionally. All 211 entries must be present;
 * missing programs are not silently replaced with a miss shader. The caller
 * owns code residency and must keep each complete entry program and every
 * transitive code/data reference alive for the submission. */
static inline bool
agx_rt_ift_driver_pack(void *out, size_t size,
                    const struct agx_rt_ift_program *programs, size_t count,
                    uint64_t *default_program, uint64_t *default_control,
                    bool gen1)
{
   if (!out || size < AGX_RT_IFT_GEN3_BYTES || !programs ||
       count != AGX_RT_IFT_GEN3_PROGRAMS ||
       !default_program || !default_control)
      return false;
   uint8_t entry[16];
   for (unsigned i = 0; i < count; ++i) {
      struct agx_rt_ift_signature s;
      agx_rt_ift_gen3_signature(i, &s);
      if (!agx_rt_ift_entry_pack(entry, programs[i].address,
                                  programs[i].resources0, programs[i].resources1,
                                  s.field3, s.field4, s.field6,
                                  !gen1 && s.type == 2))
         return false;
   }
   memset(out, 0, AGX_RT_IFT_GEN3_BYTES);
   for (unsigned i = 0; i < count; ++i) {
      struct agx_rt_ift_signature s;
      agx_rt_ift_gen3_signature(i, &s);
      agx_rt_ift_entry_pack(entry, programs[i].address,
                            programs[i].resources0, programs[i].resources1,
                            s.field3, s.field4, s.field6,
                            !gen1 && s.type == 2);
      unsigned block = agx_rt_ift_gen3_block(&s);
      for (unsigned j = 0; j <= s.additional_slots; ++j)
         memcpy((uint8_t *)out + block * 128 + (s.slot + j) * 16, entry, 16);
      if (i == 0) {
         uint64_t lo = 0, hi = 0;
         for (unsigned j = 0; j < 8; ++j) {
            lo |= (uint64_t)entry[j] << (8 * j);
            hi |= (uint64_t)entry[j + 8] << (8 * j);
         }
         *default_program = lo;
         *default_control = hi & UINT64_C(0x0000ffffffffffff);
      }
   }
   return true;
}

static inline bool
agx_rt_ift_gen3_pack(void *out, size_t size,
                    const struct agx_rt_ift_program *programs, size_t count,
                    uint64_t *default_program, uint64_t *default_control)
{
   return agx_rt_ift_driver_pack(out, size, programs, count,
                                 default_program, default_control, false);
}

/* G17P uses the same signature ordering and block indices. Its driver table
 * always sets the Gen1 entry kind, including for triangle signatures (type 2).
 * This differs from the Gen1 ordinary user-entry helper's fixed field6. */
static inline bool
agx_rt_ift_gen1_pack(void *out, size_t size,
                    const struct agx_rt_ift_program *programs, size_t count,
                    uint64_t *default_program, uint64_t *default_control)
{
   return agx_rt_ift_driver_pack(out, size, programs, count,
                                 default_program, default_control, true);
}
#endif
