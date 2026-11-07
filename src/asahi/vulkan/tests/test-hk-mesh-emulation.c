/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_mesh_emulation.h"

#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "util/macros.h"

#undef NDEBUG
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned producer_binary_size;
static unsigned pull_vs_binary_size;
static unsigned admitted_producer_binary_size;
static unsigned admitted_pull_vs_binary_size;

static bool
lower_driver_sysvals(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_geometry_param_buffer_poly:
      nir_def_replace(&intr->def,
                      nir_imm_int64(b, UINT64_C(0x1000000000)));
      return true;
   case nir_intrinsic_load_num_workgroups:
      nir_def_replace(&intr->def, nir_imm_ivec3(b, 1, 1, 1));
      return true;
   default:
      return false;
   }
}

static bool
lower_pull_vs_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   assert(sem.location == VARYING_SLOT_POS);
   b->cursor = nir_instr_remove(&intr->instr);

   u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
      nir_store_uvs_agx(
         b, nir_channel(b, intr->src[0].ssa, c),
         nir_imm_intN_t(b, nir_intrinsic_component(intr) + c, 16));
   }

   return true;
}

static nir_builder
triangle_builder(bool unsupported_varying)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &options, "G13 software mesh triangle");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 20;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   b.shader->info.outputs_written =
      VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES;
   b.shader->info.per_primitive_outputs = VARYING_BIT_PRIMITIVE_INDICES;

   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(
      &b, nir_imm_int(&b, 3), nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_store_per_vertex_output(
      &b, nir_vec4(&b, nir_i2f32(&b, lane), nir_imm_float(&b, 0.0f),
                   nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)),
      lane, nir_imm_int(&b, 0), .write_mask = 0xf,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .write_mask = 0x7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   nir_pop_if(&b, NULL);

   if (unsupported_varying) {
      b.shader->info.outputs_written |= VARYING_BIT_VAR(0);
      nir_store_per_vertex_output(
         &b, nir_imm_vec4(&b, 1.0f, 0.0f, 0.0f, 1.0f), lane,
         nir_imm_int(&b, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }

   return b;
}

static void
test_triangle(void)
{
   nir_builder b = triangle_builder(false);
   struct hk_mesh_emulation_layout layout;
   assert(hk_nir_lower_mesh_emulation(b.shader, &layout));
   nir_validate_shader(b.shader, "G13 software mesh triangle lowered");

   assert(b.shader->info.stage == MESA_SHADER_COMPUTE);
   assert(b.shader->info.outputs_written == 0);
   assert(b.shader->info.per_primitive_outputs == 0);
   assert(layout.shared_base == 64);
   assert(layout.vertex_count == 64);
   assert(layout.primitive_count == 68);
   assert(layout.position == 80);
   assert(layout.indices == 128);
   assert(layout.shared_size == 192);
   assert(b.shader->info.shared_size == layout.shared_size);
   assert(offsetof(struct hk_mesh_emulation_params, vertex_records) == 0);
   assert(offsetof(struct hk_mesh_emulation_params, primitive_records) == 8);
   assert(offsetof(struct hk_mesh_emulation_params, index_buffer) == 16);
   assert(offsetof(struct hk_mesh_emulation_params, group_records) == 24);
   assert(offsetof(struct hk_mesh_emulation_params, task_records) == 32);
   assert(offsetof(struct hk_mesh_emulation_params, draw_params) == 40);
   assert(offsetof(struct hk_mesh_emulation_params, mesh_grid) == 48);
   assert(offsetof(struct hk_mesh_emulation_params, abi_version) == 60);
   assert(sizeof(struct hk_mesh_emulation_params) == 64);
   assert(sizeof(struct hk_mesh_emulation_group_record) == 32);

   unsigned geometry_params = 0;
   unsigned global_constants = 0;
   unsigned global_stores = 0;
   unsigned zero_global_stores = 0;
   unsigned barriers = 0;
   unsigned mesh_intrinsics = 0;
   unsigned leas = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               leas += alu->op == nir_op_ulea_agx;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            geometry_params += intr->intrinsic ==
                               nir_intrinsic_load_geometry_param_buffer_poly;
            global_constants +=
               intr->intrinsic == nir_intrinsic_load_global_constant;
            global_stores += intr->intrinsic == nir_intrinsic_store_global;
            if (intr->intrinsic == nir_intrinsic_store_global) {
               const nir_const_value *value =
                  nir_src_as_const_value(intr->src[0]);
               bool all_zero = value != NULL;
               for (unsigned c = 0;
                    all_zero && c < intr->src[0].ssa->num_components; ++c) {
                  all_zero &= nir_const_value_as_uint(
                                 value[c], intr->src[0].ssa->bit_size) == 0;
               }
               zero_global_stores += all_zero;
            }
            barriers += intr->intrinsic == nir_intrinsic_barrier;
            mesh_intrinsics +=
               intr->intrinsic == nir_intrinsic_store_per_vertex_output ||
               intr->intrinsic == nir_intrinsic_store_per_primitive_output ||
               intr->intrinsic ==
                  nir_intrinsic_set_vertex_and_primitive_count;
         }
      }
   }

   assert(geometry_params == 1);
   assert(global_constants == 6);
   assert(global_stores == 8);
   /* The ABI-mismatch arm explicitly clears both group counts/bases and the
    * GPU-resident indexed-draw header, so stale allocations draw nothing. */
   assert(zero_global_stores == 2);
   assert(barriers >= 3);
   assert(mesh_intrinsics == 0);
   assert(leas == 5);

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX2,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   assert(agx_compile_shader_nir(b.shader, &key, &part));
   assert(part.binary != NULL);
   assert(part.info.binary_size > 0);
   producer_binary_size = part.info.binary_size;
   free(part.binary);
   ralloc_free(b.shader);
}

static void
test_fail_closed(void)
{
   nir_builder b = triangle_builder(true);
   assert(!hk_nir_lower_mesh_emulation(b.shader, NULL));
   assert(b.shader->info.stage == MESA_SHADER_MESH);
   assert(b.shader->info.shared_size == 20);
   assert(b.shader->info.outputs_written & VARYING_BIT_VAR(0));
   nir_validate_shader(b.shader, "unsupported mesh shader left unchanged");
   ralloc_free(b.shader);

   nir_builder returning = triangle_builder(false);
   nir_jump(&returning, nir_jump_return);
   assert(!hk_nir_lower_mesh_emulation(returning.shader, NULL));
   assert(returning.shader->info.stage == MESA_SHADER_MESH);
   nir_validate_shader(returning.shader,
                       "returning mesh shader left unchanged");
   ralloc_free(returning.shader);
}

static void
test_pull_vs(void)
{
   static const nir_shader_compiler_options options = {0};
   nir_shader *vs = hk_nir_mesh_emulation_pull_vs(&options);
   nir_validate_shader(vs, "G13 software mesh pull VS");
   assert(vs->info.stage == MESA_SHADER_VERTEX);
   assert(vs->info.io_lowered);
   assert(vs->info.outputs_written == VARYING_BIT_POS);

   unsigned geometry_params = 0;
   unsigned vertex_ids = 0;
   unsigned global_constants = 0;
   unsigned output_stores = 0;
   unsigned leas = 0;
   nir_foreach_function_impl(impl, vs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               geometry_params += intr->intrinsic ==
                                  nir_intrinsic_load_geometry_param_buffer_poly;
               vertex_ids += intr->intrinsic == nir_intrinsic_load_vertex_id;
               global_constants +=
                  intr->intrinsic == nir_intrinsic_load_global_constant;
               output_stores +=
                  intr->intrinsic == nir_intrinsic_store_output;
            } else if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               leas += alu->op == nir_op_ulea_agx;
               if (alu->op == nir_op_ulea_agx) {
                  assert(nir_src_is_const(alu->src[2].src));
                  assert(nir_src_as_uint(alu->src[2].src) == 4);
               }
            }
         }
      }
   }

   assert(geometry_params == 1);
   assert(vertex_ids == 1);
   assert(global_constants == 2);
   assert(output_stores == 1);
   assert(leas == 1);

   NIR_PASS(_, vs, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, vs, nir_shader_intrinsics_pass, lower_pull_vs_output,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(vs);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX2,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   assert(agx_compile_shader_nir(vs, &key, &part));
   assert(part.binary != NULL);
   assert(part.info.binary_size > 0);
   pull_vs_binary_size = part.info.binary_size;
   free(part.binary);
   ralloc_free(vs);
}

static unsigned
compile_producer(nir_shader *producer)
{
   NIR_PASS(_, producer, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(producer);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX2,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   assert(agx_compile_shader_nir(producer, &key, &part));
   assert(part.binary != NULL);
   assert(part.info.binary_size > 0);
   unsigned size = part.info.binary_size;
   free(part.binary);
   ralloc_free(producer);
   return size;
}

static unsigned
compile_pull_vs(nir_shader *pull_vs)
{
   NIR_PASS(_, pull_vs, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, pull_vs, nir_shader_intrinsics_pass, lower_pull_vs_output,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(pull_vs);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX2,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   assert(agx_compile_shader_nir(pull_vs, &key, &part));
   assert(part.binary != NULL);
   assert(part.info.binary_size > 0);
   unsigned size = part.info.binary_size;
   free(part.binary);
   ralloc_free(pull_vs);
   return size;
}

static void
test_admission_identity_lifetime(void)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder first = triangle_builder(false);
   nir_builder same = triangle_builder(false);
   nir_builder distinct = triangle_builder(false);
   nir_builder unsupported = triangle_builder(true);

   /* Move the contract's private shared base so the admitted programs and
    * their cache identity must differ while remaining otherwise supported. */
   distinct.shader->info.shared_size = 84;

   struct hk_mesh_emulation_admission *a =
      hk_mesh_emulation_admission_create(first.shader, &options);
   struct hk_mesh_emulation_admission *b =
      hk_mesh_emulation_admission_create(same.shader, &options);
   struct hk_mesh_emulation_admission *c =
      hk_mesh_emulation_admission_create(distinct.shader, &options);
   assert(a && b && c);
   assert(!hk_mesh_emulation_admission_create(unsupported.shader, &options));

   /* Admission clones and lowers; its caller-owned API NIR remains intact. */
   assert(first.shader->info.stage == MESA_SHADER_MESH);
   assert(same.shader->info.stage == MESA_SHADER_MESH);
   assert(distinct.shader->info.stage == MESA_SHADER_MESH);
   assert(unsupported.shader->info.stage == MESA_SHADER_MESH);
   assert(unsupported.shader->info.outputs_written & VARYING_BIT_VAR(0));

   assert(memcmp(hk_mesh_emulation_admission_identity(a),
                 hk_mesh_emulation_admission_identity(b),
                 BLAKE3_KEY_LEN) == 0);
   assert(memcmp(hk_mesh_emulation_admission_identity(a),
                 hk_mesh_emulation_admission_identity(c),
                 BLAKE3_KEY_LEN) != 0);
   assert(hk_mesh_emulation_admission_layout(a)->shared_base == 64);
   assert(hk_mesh_emulation_admission_layout(c)->shared_base == 128);

   nir_shader *producer = hk_mesh_emulation_admission_clone_producer(
      a, &options);
   nir_shader *pull_vs = hk_mesh_emulation_admission_clone_pull_vs(a, &options);
   assert(producer && pull_vs);
   assert(producer->info.stage == MESA_SHADER_COMPUTE);
   assert(pull_vs->info.stage == MESA_SHADER_VERTEX);
   admitted_producer_binary_size = compile_producer(producer);
   admitted_pull_vs_binary_size = compile_pull_vs(pull_vs);

   /* The immutable pair can be shared by cache users; its buffers disappear
    * only after the final owner releases it. */
   assert(hk_mesh_emulation_admission_ref(a) == a);
   assert(!hk_mesh_emulation_admission_unref(a));
   assert(hk_mesh_emulation_admission_unref(a));
   assert(hk_mesh_emulation_admission_unref(b));
   assert(hk_mesh_emulation_admission_unref(c));

   ralloc_free(first.shader);
   ralloc_free(same.shader);
   ralloc_free(distinct.shader);
   ralloc_free(unsupported.shader);
}

int
main(void)
{
   test_triangle();
   test_fail_closed();
   test_pull_vs();
   test_admission_identity_lifetime();
   printf("G13 mesh NIR: producer PASS (%u B), pull VS PASS (%u B), "
          "admission/cache PASS (%u/%u B), unsupported PASS\n",
          producer_binary_size, pull_vs_binary_size,
          admitted_producer_binary_size, admitted_pull_vs_binary_size);
   return 0;
}
