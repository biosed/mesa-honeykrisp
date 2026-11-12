/* SPDX-License-Identifier: MIT */
#include "agx_rt_ift.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
int main(void)
{
   /* Native descriptor for the captured rt-closest default function, with
    * the table-entry valid bit. Replayed native encoder is the oracle. */
   const uint8_t expected[16] = {0x40,0x86,0x18,0x00,0x10,0,0,0,
                                0,0,0,0,0x60,1,0,0x80};
   uint8_t out[16];
   assert(agx_rt_ift_entry_pack(out, UINT64_C(0x1000188640), 0, 0, 0, 4, 0, false));
   assert(!memcmp(out, expected, 16));
   /* G17P HAL200: ordinary Gen1 entries additionally set bit 39. */
   uint8_t gen1_expected[16];
   memcpy(gen1_expected, expected, sizeof(gen1_expected));
   gen1_expected[12] = 0xe0;
   assert(agx_rt_ift_gen1_entry_pack(out, UINT64_C(0x1000188640),
                                     0, 0, 0, 4));
   assert(!memcmp(out, gen1_expected, 16));
   memset(out, 0xa5, sizeof(out));
   assert(!agx_rt_ift_entry_pack(out, 65, 0, 0, 0, 0, 0, false));
   for (unsigned i=0; i<16; ++i) assert(out[i]==0xa5);
   assert(!agx_rt_ift_entry_pack(out, 64, 1u<<18, 0, 0, 0, 0, false));
   assert(!agx_rt_ift_entry_pack(out, 64, 0, 1u<<12, 0, 0, 0, false));
   assert(!agx_rt_ift_entry_pack(out, 64, 0, 0, 8, 0, 0, false));
   assert(!agx_rt_ift_entry_pack(out, 64, 0, 0, 0, 8, 0, false));
   assert(!agx_rt_ift_entry_pack(out, 64, 0, 0, 0, 0, 256, false));
   struct agx_rt_ift_program programs[AGX_RT_IFT_GEN3_PROGRAMS];
   uint8_t table[AGX_RT_IFT_GEN3_BYTES], occupied[AGX_RT_IFT_GEN3_BLOCKS][8] = {0};
   for (unsigned i = 0; i < AGX_RT_IFT_GEN3_PROGRAMS; ++i) {
      programs[i] = (struct agx_rt_ift_program){.address = 0x10000 + 0x100 * i};
      struct agx_rt_ift_signature sig;
      assert(agx_rt_ift_gen3_signature(i, &sig));
      unsigned block = agx_rt_ift_gen3_block(&sig);
      assert(block < AGX_RT_IFT_GEN3_BLOCKS);
      for (unsigned j = 0; j <= sig.additional_slots; ++j) {
         assert(sig.slot + j < 8 && !occupied[block][sig.slot + j]);
         occupied[block][sig.slot + j] = 1;
      }
   }
   uint64_t program = 0, control = 0;
   assert(agx_rt_ift_gen3_pack(table, sizeof(table), programs,
                              AGX_RT_IFT_GEN3_PROGRAMS, &program, &control));
   assert(program == 0x10000 && control == UINT64_C(0x16000000000));
   unsigned count = 0;
   for (unsigned b = 0; b < AGX_RT_IFT_GEN3_BLOCKS; ++b) {
      for (unsigned j = 0; j < 8; ++j) {
         uint8_t *entry = table + b * 128 + j * 16;
         count += occupied[b][j];
         if (!occupied[b][j]) {
            const uint8_t zero[16] = {0};
            assert(!memcmp(entry, zero, sizeof(zero)));
         }
      }
   }
   assert(count == 218);
   for (unsigned i = 1; i < 8; ++i)
      assert(!memcmp(table + 104 * 128, table + 104 * 128 + i * 16, 16));
   for (unsigned i = 0; i < AGX_RT_IFT_GEN3_PROGRAMS; ++i) {
      memset(table, 0xa5, sizeof(table));
      uint64_t address = programs[i].address;
      programs[i].address |= 1;
      program = control = UINT64_MAX;
      assert(!agx_rt_ift_gen3_pack(table, sizeof(table), programs,
                                  AGX_RT_IFT_GEN3_PROGRAMS, &program, &control));
      for (unsigned j = 0; j < sizeof(table); ++j) assert(table[j] == 0xa5);
      assert(program == UINT64_MAX && control == UINT64_MAX);
      programs[i].address = address;
   }
   assert(!agx_rt_ift_gen3_signature(211, NULL));
   assert(!agx_rt_ift_gen3_pack(table, sizeof(table) - 1, programs, 211,
                               &program, &control));
   assert(!agx_rt_ift_gen3_pack(table, sizeof(table), programs, 210,
                               &program, &control));
   puts("PASS native G17P entry, M5 descriptors, complete table, aliases and invalid input atomicity");
   return 0;
}
