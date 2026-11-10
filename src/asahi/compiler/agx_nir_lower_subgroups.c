/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/list.h"
#include "agx_nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_opcodes.h"

/* XXX: cribbed from nak, move to common */
static nir_def *
nir_udiv_round_up(nir_builder *b, nir_def *n, nir_def *d)
{
   return nir_udiv(b, nir_iadd(b, n, nir_iadd_imm(b, d, -1)), d);
}

struct agx_subgroup_lower_options {
   bool g17p;
   bool g17p_clustered_float;
};

static void
agx_analyze_quad_bits(nir_def *def, uint32_t *zero_out, uint32_t *qunif_out,
                      unsigned depth)
{
   *zero_out = 0;
   *qunif_out = 0;

   if (depth > 8)
      return;

   nir_instr *parent = nir_def_instr(def);
   if (parent->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_subgroup_invocation:
         *zero_out = ~0x1fu;
         *qunif_out = ~0x3u;
         return;
      case nir_intrinsic_ballot:
      case nir_intrinsic_first_invocation:
      case nir_intrinsic_last_invocation:
      case nir_intrinsic_read_first_invocation:
      case nir_intrinsic_load_subgroup_id:
      case nir_intrinsic_load_num_subgroups:
      case nir_intrinsic_load_workgroup_id:
      case nir_intrinsic_load_num_workgroups:
      case nir_intrinsic_load_push_constant:
         *zero_out = 0;
         *qunif_out = ~0u;
         return;
      default:
         return;
      }
   }

   if (def->num_components != 1)
      return;

   if (nir_def_is_const(def)) {
      uint32_t c = nir_const_value_as_uint(
         nir_instr_as_load_const(nir_def_instr(def))->value[0], def->bit_size);
      *zero_out = ~c;
      *qunif_out = ~0u;
      return;
   }

   if (parent->type != nir_instr_type_alu)
      return;

   nir_alu_instr *alu = nir_instr_as_alu(parent);
   if (alu->src[0].swizzle[0] != 0)
      return;

   uint32_t z0 = 0, q0 = 0, z1 = 0, q1 = 0;
   agx_analyze_quad_bits(alu->src[0].src.ssa, &z0, &q0, depth + 1);

   if (nir_op_infos[alu->op].num_inputs >= 2) {
      if (alu->src[1].swizzle[0] != 0)
         return;
      agx_analyze_quad_bits(alu->src[1].src.ssa, &z1, &q1, depth + 1);
   }

   switch (alu->op) {
   case nir_op_mov:
      *zero_out = z0;
      *qunif_out = q0;
      return;

   case nir_op_inot:
      *zero_out = 0;
      *qunif_out = q0;
      return;

   case nir_op_ufind_msb:
   case nir_op_ifind_msb:
   case nir_op_find_lsb:
   case nir_op_bit_count:
      if (q0 == ~0u)
         *qunif_out = ~0u;
      return;

   case nir_op_iand:
      *zero_out = z0 | z1;
      *qunif_out = *zero_out | (q0 & q1);
      return;

   case nir_op_ior:
   case nir_op_ixor:
      *zero_out = z0 & z1;
      *qunif_out = q0 & q1;
      return;

   case nir_op_ishl:
      if (nir_src_is_const(alu->src[1].src)) {
         unsigned sh = nir_src_as_uint(alu->src[1].src) & 31;
         uint32_t lo = sh ? BITFIELD_MASK(sh) : 0;
         *zero_out = (z0 << sh) | lo;
         *qunif_out = (q0 << sh) | lo;
      }
      return;

   case nir_op_ushr:
      if (nir_src_is_const(alu->src[1].src)) {
         unsigned sh = nir_src_as_uint(alu->src[1].src) & 31;
         uint32_t hi = sh ? (~0u << (32 - sh)) : 0;
         *zero_out = (z0 >> sh) | hi;
         *qunif_out = (q0 >> sh) | hi;
      }
      return;

   case nir_op_ishr:
      if (nir_src_is_const(alu->src[1].src)) {
         unsigned sh = nir_src_as_uint(alu->src[1].src) & 31;
         *zero_out = z0 >> sh;
         *qunif_out = q0 >> sh;
      }
      return;

   case nir_op_bcsel:
      if (nir_src_is_const(alu->src[0].src)) {
         if (nir_src_as_bool(alu->src[0].src)) {
            *zero_out = z1;
            *qunif_out = q1;
         } else if (alu->src[2].swizzle[0] == 0) {
            agx_analyze_quad_bits(alu->src[2].src.ssa, zero_out, qunif_out,
                                  depth + 1);
         }
      } else if (alu->src[2].swizzle[0] == 0) {
         uint32_t z2 = 0, q2 = 0;
         agx_analyze_quad_bits(alu->src[2].src.ssa, &z2, &q2, depth + 1);
         *zero_out = z1 & z2;
         *qunif_out = *zero_out | ((q0 & 1) ? (q1 & q2) : 0);
      }
      return;

   case nir_op_iadd: {
      uint32_t zero = 0, qunif = 0;
      bool carry_zero = true;
      bool carry_qunif = true;

      for (unsigned i = 0; i < 32; i++) {
         bool a_zero = (z0 >> i) & 1;
         bool b_zero = (z1 >> i) & 1;
         bool a_qunif = (q0 >> i) & 1;
         bool b_qunif = (q1 >> i) & 1;

         if (a_zero && b_zero && carry_zero)
            zero |= 1u << i;
         if (a_qunif && b_qunif && carry_qunif)
            qunif |= 1u << i;

         unsigned zero_cnt = (unsigned)a_zero + (unsigned)b_zero +
                             (unsigned)carry_zero;
         bool next_carry_zero = zero_cnt >= 2;
         bool next_carry_qunif = next_carry_zero ||
                                 (a_qunif && b_qunif && carry_qunif);
         carry_zero = next_carry_zero;
         carry_qunif = next_carry_qunif;
      }

      *zero_out = zero;
      *qunif_out = qunif;
      return;
   }

   default:
      return;
   }
}

static nir_def *
agx_chase_mov(nir_def *def)
{
   while (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op != nir_op_mov || alu->def.num_components != 1 ||
          alu->src[0].swizzle[0] != 0)
         break;
      def = alu->src[0].src.ssa;
   }
   return def;
}

static nir_def *
agx_match_quad_shuffle(nir_builder *b, nir_def *data, nir_def *target)
{
   target = agx_chase_mov(target);
   nir_instr *parent = nir_def_instr(target);

   if (parent->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(parent)->intrinsic ==
          nir_intrinsic_load_subgroup_invocation) {
      return data;
   }

   if (parent->type != nir_instr_type_alu || target->num_components != 1)
      return NULL;

   nir_alu_instr *alu = nir_instr_as_alu(parent);
   if (alu->src[0].swizzle[0] != 0 ||
       (nir_op_infos[alu->op].num_inputs >= 2 && alu->src[1].swizzle[0] != 0))
      return NULL;

   /* SHFL.BFLY within-quad XOR: ixor(load_subgroup_invocation, imm(0..3)) */
   if (alu->op == nir_op_ixor) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *a = agx_chase_mov(alu->src[side].src.ssa);
         nir_def *c = agx_chase_mov(alu->src[!side].src.ssa);
         if (nir_def_instr(a)->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(nir_def_instr(a))->intrinsic ==
                nir_intrinsic_load_subgroup_invocation &&
             nir_def_is_const(c)) {
            uint32_t mask = nir_const_value_as_uint(
                               nir_instr_as_load_const(nir_def_instr(c))->value[0],
                               c->bit_size) &
                            0x1fu;
            switch (mask) {
            case 0:
               return data;
            case 1:
               return nir_quad_swap_horizontal(b, data);
            case 2:
               return nir_quad_swap_vertical(b, data);
            case 3:
               return nir_quad_swap_diagonal(b, data);
            default:
               break;
            }
         }
      }
   }

   /* SHFL.IDX within-quad broadcast: (load_subgroup_invocation & 28) | idx */
   if (alu->op == nir_op_iand) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *a = agx_chase_mov(alu->src[side].src.ssa);
         nir_def *c = agx_chase_mov(alu->src[!side].src.ssa);
         if (nir_def_instr(a)->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(nir_def_instr(a))->intrinsic ==
                nir_intrinsic_load_subgroup_invocation &&
             nir_def_is_const(c)) {
            uint32_t mask = nir_const_value_as_uint(
                               nir_instr_as_load_const(nir_def_instr(c))->value[0],
                               c->bit_size) &
                            0x1fu;
            if (mask == 0x1cu)
               return nir_quad_broadcast(b, data, nir_imm_int(b, 0));
         }
      }
   }

   if (alu->op == nir_op_ior || alu->op == nir_op_ixor ||
       alu->op == nir_op_iadd) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *a = agx_chase_mov(alu->src[side].src.ssa);
         nir_def *c = agx_chase_mov(alu->src[!side].src.ssa);
         if (!nir_def_is_const(c) ||
             nir_def_instr(a)->type != nir_instr_type_alu)
            continue;
         uint32_t idx = nir_const_value_as_uint(
            nir_instr_as_load_const(nir_def_instr(c))->value[0], c->bit_size);
         if (idx > 3)
            continue;
         nir_alu_instr *and_alu = nir_instr_as_alu(nir_def_instr(a));
         if (and_alu->op != nir_op_iand || and_alu->src[0].swizzle[0] != 0 ||
             and_alu->src[1].swizzle[0] != 0)
            continue;
         for (unsigned s2 = 0; s2 < 2; s2++) {
            nir_def *la = agx_chase_mov(and_alu->src[s2].src.ssa);
            nir_def *lc = agx_chase_mov(and_alu->src[!s2].src.ssa);
            if (nir_def_instr(la)->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(nir_def_instr(la))->intrinsic ==
                   nir_intrinsic_load_subgroup_invocation &&
                nir_def_is_const(lc)) {
               uint32_t mask =
                  nir_const_value_as_uint(
                     nir_instr_as_load_const(nir_def_instr(lc))->value[0],
                     lc->bit_size) &
                  0x1fu;
               if (mask == 0x1cu)
                  return nir_quad_broadcast(b, data, nir_imm_int(b, idx));
            }
         }
      }
   }

   return NULL;
}

/* Returns a 2-bit mask of within-quad XOR symmetries of `def`:
 * bit 0 (1): invariant under lane ^ 1 (horizontal pair equal)
 * bit 1 (2): invariant under lane ^ 2 (vertical pair equal)
 * bits 0|1 (3): quad-uniform (all 4 lanes of each quad hold the same value)
 */
static unsigned
agx_get_quad_symmetry(nir_def *def, unsigned depth)
{
   def = agx_chase_mov(def);
   if (depth > 8 || def->num_components != 1)
      return 0;

   if (nir_def_is_const(def))
      return 3;

   nir_instr *parent = nir_def_instr(def);
   if (parent->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent);
      switch (intr->intrinsic) {
      case nir_intrinsic_quad_broadcast:
         return nir_src_is_const(intr->src[1]) ? 3 : 0;
      case nir_intrinsic_read_invocation: {
         uint32_t z = 0, q = 0;
         agx_analyze_quad_bits(intr->src[1].ssa, &z, &q, 0);
         return (q & 0x1fu) == 0x1fu ? 3 : 0;
      }
      case nir_intrinsic_quad_swap_horizontal:
      case nir_intrinsic_quad_swap_vertical:
      case nir_intrinsic_quad_swap_diagonal:
         return agx_get_quad_symmetry(intr->src[0].ssa, depth + 1);
      default: {
         uint32_t z = 0, q = 0;
         agx_analyze_quad_bits(def, &z, &q, 0);
         return (q == ~0u) ? 3 : 0;
      }
      }
   }

   if (parent->type != nir_instr_type_alu)
      return 0;

   nir_alu_instr *alu = nir_instr_as_alu(parent);
   unsigned num_inputs = nir_op_infos[alu->op].num_inputs;
   for (unsigned i = 0; i < num_inputs; i++) {
      if (alu->src[i].swizzle[0] != 0)
         return 0;
   }

   if (alu->op == nir_op_bcsel && nir_src_is_const(alu->src[0].src)) {
      nir_def *chosen = nir_src_as_bool(alu->src[0].src)
                           ? alu->src[1].src.ssa
                           : alu->src[2].src.ssa;
      return agx_get_quad_symmetry(chosen, depth + 1);
   }

   unsigned sym = 3;
   for (unsigned i = 0; i < num_inputs; i++)
      sym &= agx_get_quad_symmetry(alu->src[i].src.ssa, depth + 1);

   if (num_inputs == 2 &&
       (nir_op_infos[alu->op].algebraic_properties &
        NIR_OP_IS_2SRC_COMMUTATIVE)) {
      nir_def *s0 = agx_chase_mov(alu->src[0].src.ssa);
      nir_def *s1 = agx_chase_mov(alu->src[1].src.ssa);
      for (unsigned side = 0; side < 2; side++) {
         nir_def *a = side ? s1 : s0;
         nir_def *b = side ? s0 : s1;
         nir_instr *b_instr = nir_def_instr(b);
         if (b_instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *b_intr = nir_instr_as_intrinsic(b_instr);
            nir_def *b_src =
               (b_intr->intrinsic == nir_intrinsic_quad_swap_horizontal ||
                b_intr->intrinsic == nir_intrinsic_quad_swap_vertical ||
                b_intr->intrinsic == nir_intrinsic_quad_swap_diagonal)
                  ? agx_chase_mov(b_intr->src[0].ssa)
                  : NULL;
            if (b_src == a) {
               unsigned a_sym = agx_get_quad_symmetry(a, depth + 1);
               if (b_intr->intrinsic == nir_intrinsic_quad_swap_horizontal)
                  sym |= a_sym | 1;
               else if (b_intr->intrinsic == nir_intrinsic_quad_swap_vertical)
                  sym |= a_sym | 2;
               else if (b_intr->intrinsic == nir_intrinsic_quad_swap_diagonal &&
                        (a_sym == 1 || a_sym == 2))
                  sym |= 3;
            }
         }
      }
   }

   return sym;
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_quad_vote_any: {
      nir_def *ballot = nir_quad_ballot_agx(b, 16, intr->src[0].ssa);
      nir_def_rewrite_uses(&intr->def, nir_ine_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_quad_vote_all: {
      nir_def *ballot =
         nir_quad_ballot_agx(b, 16, nir_inot(b, intr->src[0].ssa));
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_elect: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, active_id, 0));
      return true;
   }

   case nir_intrinsic_first_invocation: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def *is_first = nir_ieq_imm(b, active_id, 0);
      nir_def *first_bit = nir_ballot(b, 1, 32, is_first);
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, first_bit));
      return true;
   }

   case nir_intrinsic_last_invocation: {
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, active_mask));
      return true;
   }

   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq: {
      /* The common lowering does:
       *
       *    vote_all(x == read_first(x))
       *
       * This is not optimal for AGX, since we have ufind_msb but not ctz, so
       * it's cheaper to read the last invocation than the first. So we do:
       *
       *    vote_all(x == read_last(x))
       *
       * implemented with lowered instructions as
       *
       *    ballot(x != broadcast(x, ffs(ballot(true)))) == 0
       */
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *active_bit = nir_ufind_msb(b, active_mask);
      nir_def *other = nir_read_invocation(b, intr->src[0].ssa, active_bit);
      nir_def *is_ne;

      if (intr->intrinsic == nir_intrinsic_vote_feq) {
         is_ne = nir_fneu(b, other, intr->src[0].ssa);
      } else {
         is_ne = nir_ine(b, other, intr->src[0].ssa);
      }

      nir_def *ballot = nir_ballot(b, 1, 32, is_ne);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_load_num_subgroups: {
      nir_def *workgroup_size = nir_load_workgroup_size(b);
      workgroup_size = nir_imul(b,
                                nir_imul(b, nir_channel(b, workgroup_size, 0),
                                         nir_channel(b, workgroup_size, 1)),
                                nir_channel(b, workgroup_size, 2));
      nir_def *subgroup_size = nir_imm_int(b, 32);
      nir_def *num_subgroups =
         nir_udiv_round_up(b, workgroup_size, subgroup_size);
      nir_def_rewrite_uses(&intr->def, num_subgroups);
      return true;
   }

   case nir_intrinsic_shuffle: {
      nir_def *data = intr->src[0].ssa;
      nir_def *target = intr->src[1].ssa;

      /* The hardware shuffle instruction chooses a single index within the
       * target quad to shuffle each source quad with. Consequently, the low
       * 2-bits of shuffle indices should not be quad divergent.  When one or
       * both low bits are already quad-uniform, only emit the required 1 or 2
       * hardware shuffles instead of all 4.
       */
      if (b->shader->info.stage != MESA_SHADER_COMPUTE ||
          !b->shader->info.cs.has_cooperative_matrix) {
         nir_def *quad_shuf = agx_match_quad_shuffle(b, data, target);
         if (quad_shuf) {
            nir_def_rewrite_uses(&intr->def, quad_shuf);
            return true;
         }

         uint32_t zero_mask = 0, qunif_mask = 0;
         agx_analyze_quad_bits(target, &zero_mask, &qunif_mask, 0);

         if ((qunif_mask & 3) == 3) {
            nir_def *lane = (zero_mask & ~0x1fu) == ~0x1fu
                               ? target
                               : nir_iand_imm(b, target, 0x1f);
            nir_def_rewrite_uses(&intr->def, nir_read_invocation(b, data, lane));
            return true;
         }

         /* If `data` is already quad-uniform (all 4 lanes of each quad hold
          * the same value, e.g. after butterfly steps ^1 and ^2) and `target`
          * selects the same quad across all 4 lanes (`(qunif_mask & 0x1c) ==
          * 0x1c`, e.g. butterfly steps ^4, ^8, ^16), any lane in the target
          * quad holds the required value.
          */
         if ((qunif_mask & 0x1cu) == 0x1cu &&
             agx_get_quad_symmetry(data, 0) == 3) {
            nir_def *lane = nir_iand_imm(b, target, 0x1c);
            nir_def_rewrite_uses(&intr->def, nir_read_invocation(b, data, lane));
            return true;
         }

         if ((qunif_mask & 3) == 1 || (qunif_mask & 3) == 2) {
            unsigned bit = (qunif_mask & 3) == 1 ? 2 : 1;
            nir_def *target_0 = nir_iand_imm(b, target, 0x1f & ~bit);
            nir_def *target_1 = nir_iadd_imm(b, target_0, bit);
            nir_def *shuf_0 = nir_read_invocation(b, data, target_0);
            nir_def *shuf_1 = nir_read_invocation(b, data, target_1);
            nir_def *result =
               nir_bcsel(b, nir_ieq(b, target, target_1), shuf_1, shuf_0);
            nir_def_rewrite_uses(&intr->def, result);
            return true;
         }
      }

      nir_def *quad_start = nir_iand_imm(b, target, 0x1c);
      nir_def *result = NULL;

      for (unsigned i = 0; i < 4; ++i) {
         nir_def *target_i = nir_iadd_imm(b, quad_start, i);
         nir_def *shuf = nir_read_invocation(b, data, target_i);

         if (result)
            result = nir_bcsel(b, nir_ieq(b, target, target_i), shuf, result);
         else
            result = shuf;
      }

      nir_def_rewrite_uses(&intr->def, result);
      return true;
   }

   case nir_intrinsic_inclusive_scan: {
      /* If we got here, we support the corresponding exclusive scan in
       * hardware, so just handle the last element.
       */
      nir_op red_op = nir_intrinsic_reduction_op(intr);
      nir_def *data = intr->src[0].ssa;

      b->cursor = nir_after_instr(&intr->instr);
      intr->intrinsic = nir_intrinsic_exclusive_scan;
      nir_def *accum = nir_build_alu2(b, red_op, data, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, accum);
      return true;
   }

   case nir_intrinsic_ballot: {
      /* Optimize popcount(ballot(true)) to load_active_subgroup_count_agx().
       *
       * Not on G17P: Apple's compiler reads its active-count register (SR36)
       * only inside the helper-guarded region of a uniform atomic, and
       * compiles simd_sum(1u) to a reduction instead, so keep the ballot whose
       * lane membership every other subgroup operation shares.
       */
      const struct agx_subgroup_lower_options *agx_options = data;
      if (agx_options->g17p || !nir_src_is_const(intr->src[0]) ||
          !nir_src_as_bool(intr->src[0]) ||
          !list_is_singular(&intr->def.uses))
         return false;

      nir_src *use = list_first_entry(&intr->def.uses, nir_src, use_link);
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->op != nir_op_bit_count)
         return false;

      nir_def_rewrite_uses(&alu->def,
                           nir_load_active_subgroup_count_agx(b, 32));
      return true;
   }

   default:
      return false;
   }
}



static bool
lower_subgroup_filter(const nir_intrinsic_instr *intr, const void *data)
{
   const struct agx_subgroup_lower_options *agx_options = data;
   unsigned cluster = 0;

   /* Use default behaviour for everything but scans */
   if (intr->intrinsic != nir_intrinsic_exclusive_scan &&
       intr->intrinsic != nir_intrinsic_inclusive_scan &&
       intr->intrinsic != nir_intrinsic_reduce)
      return true;

   if (intr->def.num_components > 1 || intr->def.bit_size == 1)
      return true;

   /* Hardware supports quad ops but no other support clustered reductions. */
   if (nir_intrinsic_has_cluster_size(intr)) {
      cluster = nir_intrinsic_cluster_size(intr);
      if (cluster && cluster != 4 && cluster < 32)
         return true;
   }

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_imul:
      return true;

   case nir_op_fmin:
   case nir_op_fmax:
      /* Fresh G17P witnesses pin the native qRedfmin/qRedfmax forms. Keep
       * every other target and non-quad shape on the established common
       * lowering path. */
      if (agx_options->g17p_clustered_float && cluster == 4 &&
          intr->def.bit_size == 32)
         return false;
      return true;

   case nir_op_iadd:
   case nir_op_iand:
   case nir_op_ixor:
   case nir_op_ior:
      /* these have dedicated 64-bit lowering paths that use the 32-bit hardware
       * instructions so are likely better than the full lowering.
       */
      return false;

   default:
      /* otherwise, lower 64-bit, since the hw ops are at most 32-bit. */
      return intr->def.bit_size == 64;
   }
}

static bool
agx_match_redux_shuffle(nir_def *def, unsigned expected_lane,
                        nir_block *block, nir_def **src_out)
{
   def = agx_chase_mov(def);
   if (nir_def_instr(def)->type != nir_instr_type_intrinsic ||
       nir_def_instr(def)->block != block)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(def));
   if (intr->intrinsic != nir_intrinsic_shuffle)
      return false;

   nir_def *lane = agx_chase_mov(intr->src[1].ssa);
   if (!nir_def_is_const(lane))
      return false;

   uint32_t lane_idx = nir_const_value_as_uint(
      nir_instr_as_load_const(nir_def_instr(lane))->value[0], lane->bit_size);
   if (lane_idx != expected_lane)
      return false;

   nir_def *src = agx_chase_mov(intr->src[0].ssa);
   if (*src_out == NULL)
      *src_out = src;
   return *src_out == src;
}

static bool
agx_is_ballot_true_expr(nir_def *def, nir_block *block,
                        nir_def **ballot_out, unsigned depth)
{
   def = agx_chase_mov(def);
   if (depth > 8)
      return false;

   if (nir_def_is_const(def))
      return true;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_ballot || instr->block != block ||
          !nir_src_is_const(intr->src[0]) || !nir_src_as_bool(intr->src[0]))
         return false;
      if (*ballot_out == NULL)
         *ballot_out = &intr->def;
      return *ballot_out == &intr->def;
   }

   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   unsigned n = nir_op_infos[alu->op].num_inputs;
   for (unsigned i = 0; i < n; i++) {
      if (!agx_is_ballot_true_expr(alu->src[i].src.ssa, block, ballot_out,
                                   depth + 1))
         return false;
   }
   return true;
}

/* Recognize the 32-lane unrolled REDUX pattern (ballot(true) + 32
 * shuffle(src, 0..31) + bcsel + associative reduction op) and fold it into a
 * single hardware subgroup reduction (`nir_intrinsic_reduce`).
 */
static bool
agx_opt_redux(nir_shader *s)
{
   bool progress = false;

   nir_foreach_function_impl(impl, s) {
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op != nir_op_bcsel || alu->def.num_components != 1 ||
                alu->def.bit_size != 32)
               continue;

            nir_def *cur_def = &alu->def;
            nir_def *src = NULL;
            nir_def *ballot_def = NULL;
            nir_op red_op = nir_num_opcodes;
            bool matched = true;

            for (int lane = 31; lane >= 1; lane--) {
               cur_def = agx_chase_mov(cur_def);
               if (nir_def_instr(cur_def)->type != nir_instr_type_alu ||
                   nir_def_instr(cur_def)->block != block) {
                  matched = false;
                  break;
               }
               nir_alu_instr *sel = nir_instr_as_alu(nir_def_instr(cur_def));
               if (sel->op != nir_op_bcsel || sel->def.num_components != 1 ||
                   !agx_is_ballot_true_expr(sel->src[0].src.ssa, block,
                                            &ballot_def, 0)) {
                  matched = false;
                  break;
               }

               nir_def *prev_acc = agx_chase_mov(sel->src[2].src.ssa);
               nir_def *r_def = agx_chase_mov(sel->src[1].src.ssa);
               if (nir_def_instr(r_def)->type != nir_instr_type_alu ||
                   nir_def_instr(r_def)->block != block) {
                  matched = false;
                  break;
               }

               nir_alu_instr *r_alu = nir_instr_as_alu(nir_def_instr(r_def));
               if (lane == 31) {
                  switch (r_alu->op) {
                  case nir_op_iand:
                  case nir_op_ior:
                  case nir_op_ixor:
                  case nir_op_iadd:
                  case nir_op_imin:
                  case nir_op_umin:
                  case nir_op_imax:
                  case nir_op_umax:
                     red_op = r_alu->op;
                     break;
                  default:
                     matched = false;
                     break;
                  }
                  if (!matched)
                     break;
               } else if (r_alu->op != red_op) {
                  matched = false;
                  break;
               }

               nir_def *r0 = agx_chase_mov(r_alu->src[0].src.ssa);
               nir_def *r1 = agx_chase_mov(r_alu->src[1].src.ssa);
               nir_def *shfl_def = (r0 == prev_acc) ? r1
                                 : (r1 == prev_acc) ? r0
                                                    : NULL;
               if (!shfl_def ||
                   !agx_match_redux_shuffle(shfl_def, (unsigned)lane, block,
                                            &src)) {
                  matched = false;
                  break;
               }

               cur_def = prev_acc;
            }

            if (!matched || !ballot_def || !src)
               continue;

            cur_def = agx_chase_mov(cur_def);
            if (nir_def_instr(cur_def)->type != nir_instr_type_alu ||
                nir_def_instr(cur_def)->block != block)
               continue;

            nir_alu_instr *sel0 = nir_instr_as_alu(nir_def_instr(cur_def));
            if (sel0->op != nir_op_bcsel || sel0->def.num_components != 1 ||
                !agx_is_ballot_true_expr(sel0->src[0].src.ssa, block,
                                         &ballot_def, 0) ||
                !nir_src_is_const(sel0->src[2].src) ||
                !agx_match_redux_shuffle(sel0->src[1].src.ssa, 0, block, &src))
               continue;

            b.cursor = nir_before_instr(&alu->instr);
            nir_def *red =
               nir_reduce(&b, src, .reduction_op = red_op, .cluster_size = 0);
            nir_def_replace(&alu->def, red);
            progress = true;
         }
      }
   }

   return progress;
}

static bool
agx_cf_node_has_return_or_halt(nir_cf_node *node)
{
   nir_foreach_block_in_cf_node(block, node) {
      nir_instr *last = nir_block_last_instr(block);
      if (last && last->type == nir_instr_type_jump) {
         nir_jump_type t = nir_instr_as_jump(last)->type;
         if (t == nir_jump_return || t == nir_jump_halt)
            return true;
      }
   }
   return false;
}

/* In compute/task/mesh shaders whose workgroup size is fixed and a multiple of
 * 32, all 32 lanes of every subgroup are active in top-level blocks prior to
 * any return/halt. Fold `ballot(true)` in those blocks to `0xffffffff` so dead
 * partial-subgroup fallback branches can be eliminated by `nir_opt_dead_cf`.
 */
static bool
agx_opt_full_subgroup_ballot(nir_shader *s)
{
   if (!mesa_shader_stage_uses_workgroup(s->info.stage) ||
       s->info.workgroup_size_variable)
      return false;

   uint32_t wg_size = (uint32_t)s->info.workgroup_size[0] *
                      (uint32_t)s->info.workgroup_size[1] *
                      (uint32_t)s->info.workgroup_size[2];
   if (wg_size == 0 || (wg_size % 32) != 0)
      return false;

   bool progress = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!impl)
      return false;

   nir_builder b = nir_builder_create(impl);
   foreach_list_typed(nir_cf_node, cf_node, node, &impl->body) {
      if (cf_node->type != nir_cf_node_block) {
         if (agx_cf_node_has_return_or_halt(cf_node))
            break;
         continue;
      }

      nir_block *block = nir_cf_node_as_block(cf_node);
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_ballot &&
             intr->def.num_components == 1 && intr->def.bit_size == 32 &&
             nir_src_is_const(intr->src[0]) && nir_src_as_bool(intr->src[0])) {
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&intr->def, nir_imm_int(&b, -1));
            progress = true;
         }
      }

      if (agx_cf_node_has_return_or_halt(cf_node))
         break;
   }

   if (progress)
      nir_progress(true, impl, nir_metadata_control_flow);

   return progress;
}

static bool
agx_nir_lower_subgroups_internal(nir_shader *s, bool g17p_clustered_float)
{
   const struct agx_subgroup_lower_options agx_options = {
      .g17p = g17p_clustered_float,
      .g17p_clustered_float = g17p_clustered_float,
   };

   /* First, do as much common lowering as we can */
   nir_lower_subgroups_options opts = {
      .filter = lower_subgroup_filter,
      .filter_data = &agx_options,
      .lower_read_first_invocation = true,
      .lower_inverse_ballot = true,
      .lower_to_scalar = true,
      .lower_relative_shuffle = true,
      .lower_rotate_to_shuffle = true,
      .lower_subgroup_masks = true,
      .lower_reduce = true,
      .lower_vote = true,
      .ballot_components = 1,
      .ballot_bit_size = 32,
      .subgroup_size = 32,
   };

   bool progress = false;
   if (!g17p_clustered_float)
      NIR_PASS(progress, s, nir_opt_copy_prop);

   progress |= nir_lower_subgroups(s, &opts);

   if (!g17p_clustered_float &&
       (s->info.stage != MESA_SHADER_COMPUTE ||
        !s->info.cs.has_cooperative_matrix)) {
      NIR_PASS(progress, s, nir_opt_copy_prop);
      if (agx_opt_redux(s)) {
         progress = true;
         NIR_PASS(_, s, nir_opt_dce);
      }
      if (agx_opt_full_subgroup_ballot(s))
         progress = true;
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_uub, &(nir_opt_uub_options){0});
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_opt_dce);
   }

   /* Then do AGX-only lowerings on top */
   progress |= nir_shader_intrinsics_pass(s, lower, nir_metadata_control_flow,
                                          (void *)&agx_options);

   return progress;
}

bool
agx_nir_lower_subgroups(nir_shader *s)
{
   return agx_nir_lower_subgroups_internal(s, false);
}

bool
agx_nir_lower_subgroups_g17p(nir_shader *s)
{
   return agx_nir_lower_subgroups_internal(s, true);
}
