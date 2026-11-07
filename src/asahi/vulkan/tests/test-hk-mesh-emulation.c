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
static unsigned wide_producer_binary_size;
static unsigned wide_pull_vs_binary_size;
static unsigned limit_producer_binary_size;
static unsigned limit_pull_vs_binary_size;
static unsigned task_producer_binary_size;
static unsigned task_mesh_binary_size;

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
   assert(sem.location == VARYING_SLOT_POS ||
          (sem.location >= VARYING_SLOT_VAR0 &&
           sem.location <= VARYING_SLOT_VAR31));
   b->cursor = nir_instr_remove(&intr->instr);

   const unsigned base = sem.location == VARYING_SLOT_POS
      ? 0 : 4 + 4 * (sem.location - VARYING_SLOT_VAR0);

   u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
      nir_store_uvs_agx(
         b, nir_channel(b, intr->src[0].ssa, c),
         nir_imm_intN_t(b, base + nir_intrinsic_component(intr) + c, 16));
   }

   return true;
}

static nir_builder
triangle_builder(unsigned vertex_slots, unsigned primitive_slots)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "G13 software mesh triangle");
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

   for (unsigned i = 0; i < vertex_slots; ++i) {
      const unsigned slot = VARYING_SLOT_VAR0 + i;
      b.shader->info.outputs_written |= BITFIELD64_BIT(slot);
      nir_store_per_vertex_output(
         &b, nir_imm_vec4(&b, (float)i, 0.0f, 0.0f, 1.0f), lane,
         nir_imm_int(&b, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = slot, .num_slots = 1});
   }

   for (unsigned i = 0; i < primitive_slots; ++i) {
      const unsigned slot = VARYING_SLOT_VAR0 + vertex_slots + i;
      b.shader->info.outputs_written |= BITFIELD64_BIT(slot);
      b.shader->info.per_primitive_outputs |= BITFIELD64_BIT(slot);
      nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      nir_store_per_primitive_output(
         &b, nir_imm_vec4(&b, 0.0f, (float)i, 0.0f, 1.0f),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 0xf,
         .src_type = nir_type_float32,
         .io_semantics = {.location = slot, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }

   return b;
}

static void
test_triangle(void)
{
   nir_builder b = triangle_builder(1, 1);
   struct hk_mesh_emulation_layout layout;
   assert(hk_nir_lower_mesh_emulation(b.shader, &layout));
   nir_validate_shader(b.shader, "G13 software mesh triangle lowered");

   assert(b.shader->info.stage == MESA_SHADER_COMPUTE);
   assert(b.shader->info.outputs_written == 0);
   assert(b.shader->info.per_primitive_outputs == 0);
   assert(layout.vertex_count == 24);
   assert(layout.primitive_count == 28);
   assert(layout.shared_size == 32);
   assert(layout.vertex_stride_B == 32);
   assert(layout.primitive_stride_B == 32);
   assert(layout.expanded_stride_B == 48);
   assert(layout.position_offset_B == 0);
   assert(layout.index_offset_B == 0);
   assert(layout.vertex_offset_B[VARYING_SLOT_VAR0] == 16);
   assert(layout.primitive_offset_B[VARYING_SLOT_VAR1] == 16);
   assert(layout.expanded_offset_B[VARYING_SLOT_VAR0] == 16);
   assert(layout.expanded_offset_B[VARYING_SLOT_VAR1] == 32);
   assert(layout.vertex_outputs ==
          (VARYING_BIT_POS | VARYING_BIT_VAR(0)));
   assert(layout.primitive_outputs ==
          (VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_VAR(1)));
   assert(b.shader->info.shared_size == layout.shared_size);
   assert(offsetof(struct hk_mesh_emulation_params, vertex_records) == 0);
   assert(offsetof(struct hk_mesh_emulation_params, primitive_records) == 8);
   assert(offsetof(struct hk_mesh_emulation_params, expanded_records) == 16);
   assert(offsetof(struct hk_mesh_emulation_params, index_buffer) == 24);
   assert(offsetof(struct hk_mesh_emulation_params, group_records) == 32);
   assert(offsetof(struct hk_mesh_emulation_params, task_records) == 40);
   assert(offsetof(struct hk_mesh_emulation_params, draw_params) == 48);
   assert(offsetof(struct hk_mesh_emulation_params, group_to_task) == 56);
   assert(offsetof(struct hk_mesh_emulation_params, mesh_grid) == 64);
   assert(offsetof(struct hk_mesh_emulation_params, task_grid) == 76);
   assert(offsetof(struct hk_mesh_emulation_params, task_record_stride_B) ==
          88);
   assert(offsetof(struct hk_mesh_emulation_params, task_payload_size_B) ==
          92);
   assert(offsetof(struct hk_mesh_emulation_params, abi_version) == 96);
   assert(offsetof(struct hk_mesh_emulation_params, flags) == 100);
   assert(offsetof(struct hk_mesh_emulation_params, task_emitted_groups) ==
          104);
   assert(offsetof(struct hk_mesh_emulation_params, task_overflow) == 108);
   assert(offsetof(struct hk_mesh_emulation_params, reserved) == 112);
   assert(sizeof(struct hk_mesh_emulation_params) == 128);
   assert(sizeof(struct hk_mesh_emulation_group_record) == 32);
   assert(sizeof(struct hk_mesh_emulation_task_record) == 16);
   assert(sizeof(struct hk_mesh_emulation_group_task) == 8);

   unsigned geometry_params = 0;
   unsigned global_constants = 0;
   unsigned global_stores = 0;
   unsigned zero_global_stores = 0;
   unsigned barriers = 0;
   unsigned mesh_intrinsics = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu)
               continue;
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

   assert(geometry_params >= 1);
   assert(global_constants >= 6);
   assert(global_stores >= 8);
   /* The ABI-mismatch arm explicitly clears both group counts/bases and the
    * GPU-resident indexed-draw header, so stale allocations draw nothing. */
   assert(zero_global_stores == 2);
   assert(barriers >= 3);
   assert(mesh_intrinsics == 0);

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {
         .chip = AGX_CHIP_G13G,
         .encoding = AGX_ISA_ENCODING_AGX2,
         .needs_g13x_coherency = U_TRISTATE_NO,
      },
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
   nir_builder b = triangle_builder(0, 0);
   b.shader->info.outputs_written |= VARYING_BIT_PSIZ;
   nir_store_per_vertex_output(
      &b, nir_imm_float(&b, 1.0f), nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});
   assert(!hk_nir_lower_mesh_emulation(b.shader, NULL));
   assert(b.shader->info.stage == MESA_SHADER_MESH);
   assert(b.shader->info.shared_size == 20);
   assert(b.shader->info.outputs_written & VARYING_BIT_PSIZ);
   nir_validate_shader(b.shader, "unsupported mesh shader left unchanged");
   ralloc_free(b.shader);

   nir_builder returning = triangle_builder(0, 0);
   nir_jump(&returning, nir_jump_return);
   assert(!hk_nir_lower_mesh_emulation(returning.shader, NULL));
   assert(returning.shader->info.stage == MESA_SHADER_MESH);
   nir_validate_shader(returning.shader,
                       "returning mesh shader left unchanged");
   ralloc_free(returning.shader);

   nir_builder oversized_z = triangle_builder(0, 0);
   oversized_z.shader->info.workgroup_size[0] = 1;
   oversized_z.shader->info.workgroup_size[2] =
      HK_MESH_EMULATION_MAX_WORKGROUP_Z + 1;
   assert(!hk_nir_lower_mesh_emulation(oversized_z.shader, NULL));
   assert(oversized_z.shader->info.stage == MESA_SHADER_MESH);
   ralloc_free(oversized_z.shader);

   nir_builder task_payload = triangle_builder(0, 0);
   task_payload.shader->info.task_payload_size =
      HK_MESH_EMULATION_MAX_TASK_PAYLOAD + 1;
   assert(!hk_nir_lower_mesh_emulation(task_payload.shader, NULL));
   assert(task_payload.shader->info.stage == MESA_SHADER_MESH);
   ralloc_free(task_payload.shader);

   nir_builder narrow = triangle_builder(0, 0);
   narrow.shader->info.outputs_written |= VARYING_BIT_VAR(0);
   nir_store_per_vertex_output(
      &narrow, nir_imm_intN_t(&narrow, 1, 16), nir_imm_int(&narrow, 0),
      nir_imm_int(&narrow, 0), .write_mask = 1, .src_type = nir_type_uint16,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   assert(!hk_nir_lower_mesh_emulation(narrow.shader, NULL));
   assert(narrow.shader->info.stage == MESA_SHADER_MESH);
   ralloc_free(narrow.shader);

   nir_builder bad_indices = triangle_builder(0, 0);
   nir_store_per_primitive_output(
      &bad_indices, nir_imm_ivec3(&bad_indices, 0, 1, 2),
      nir_imm_int(&bad_indices, 0), nir_imm_int(&bad_indices, 0),
      .write_mask = 0x7, .src_type = nir_type_int32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   assert(!hk_nir_lower_mesh_emulation(bad_indices.shader, NULL));
   ralloc_free(bad_indices.shader);

   nir_builder mixed_type = triangle_builder(1, 0);
   nir_store_per_vertex_output(
      &mixed_type, nir_imm_ivec4(&mixed_type, 1, 2, 3, 4),
      nir_imm_int(&mixed_type, 0), nir_imm_int(&mixed_type, 0),
      .write_mask = 0xf, .src_type = nir_type_int32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   assert(!hk_nir_lower_mesh_emulation(mixed_type.shader, NULL));
   ralloc_free(mixed_type.shader);
}

static void
test_partial_components_and_types(void)
{
   nir_builder b = triangle_builder(0, 0);
   b.shader->info.outputs_written |=
      VARYING_BIT_VAR(0) | VARYING_BIT_VAR(1);
   b.shader->info.per_primitive_outputs |= VARYING_BIT_VAR(1);
   nir_store_per_vertex_output(
      &b, nir_imm_ivec2(&b, 7, 9), nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .write_mask = 0x3, .component = 1,
      .src_type = nir_type_int32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_store_per_primitive_output(
      &b, nir_imm_int(&b, 11), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .write_mask = 1, .component = 2, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});

   struct hk_mesh_emulation_layout layout;
   assert(hk_nir_lower_mesh_emulation(b.shader, &layout));
   assert(layout.vertex_components[VARYING_SLOT_VAR0] == 3);
   assert(layout.primitive_components[VARYING_SLOT_VAR1] == 3);
   assert(layout.output_type[VARYING_SLOT_VAR0] == nir_type_int32);
   assert(layout.output_type[VARYING_SLOT_VAR1] == nir_type_uint32);
   assert(layout.vertex_stride_B == 32);
   assert(layout.primitive_stride_B == 32);
   assert(layout.expanded_stride_B == 48);
   nir_validate_shader(b.shader, "partial typed mesh outputs lowered");
   ralloc_free(b.shader);
}

static void
test_pull_vs(void)
{
   nir_builder mesh = triangle_builder(1, 1);
   struct hk_mesh_emulation_layout layout;
   assert(hk_nir_lower_mesh_emulation(mesh.shader, &layout));
   nir_shader *vs = hk_nir_mesh_emulation_pull_vs(&agx_nir_options, &layout);
   nir_validate_shader(vs, "G13 software mesh pull VS");
   assert(vs->info.stage == MESA_SHADER_VERTEX);
   assert(vs->info.io_lowered);
   assert(vs->info.outputs_written ==
          (VARYING_BIT_POS | VARYING_BIT_VAR(0) | VARYING_BIT_VAR(1)));

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
                  assert(nir_src_as_uint(alu->src[2].src) == 0);
               }
            }
         }
      }
   }

   assert(geometry_params == 1);
   assert(vertex_ids == 1);
   assert(global_constants == 4);
   assert(output_stores == 3);
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
   ralloc_free(mesh.shader);
}

static unsigned
compile_producer(nir_shader *producer)
{
   NIR_PASS(_, producer, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(producer);
   struct agx_shader_key key = {
      .dev = {
         .chip = AGX_CHIP_G13G,
         .encoding = AGX_ISA_ENCODING_AGX2,
         .needs_g13x_coherency = U_TRISTATE_NO,
      },
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
test_title_inventory_width(void)
{
   /* The 487-shader title inventory reaches 56 f32 user components, location
    * 19, and 37,376 bytes of declared output.  Use 12 vec4 plus eight scalar
    * locations (56 total across VAR0..19) at the observed 160V/128P maxima.
    * This intentionally exceeds the old 32-KiB contract while remaining
    * below ABI-v3's explicit 48-KiB cap. */
   nir_builder b = triangle_builder(12, 0);
   nir_def *lane = nir_load_local_invocation_index(&b);
   for (unsigned i = 12; i < 20; ++i) {
      const unsigned slot = VARYING_SLOT_VAR0 + i;
      b.shader->info.outputs_written |= BITFIELD64_BIT(slot);
      nir_store_per_vertex_output(
         &b, nir_imm_float(&b, (float)i), lane, nir_imm_int(&b, 0),
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = slot, .num_slots = 1});
   }
   b.shader->info.mesh.max_vertices_out = 160;
   b.shader->info.mesh.max_primitives_out = 128;

   struct hk_mesh_emulation_layout layout;
   assert(hk_nir_lower_mesh_emulation(b.shader, &layout));
   assert(layout.vertex_stride_B == 240);
   assert(layout.primitive_stride_B == 16);
   assert(layout.expanded_stride_B == 240);
   assert(layout.output_memory_B == 40448);
   assert(layout.output_memory_B > 37376);
   assert(layout.output_memory_B <= 48 * 1024);
   for (unsigned i = 0; i < 12; ++i)
      assert(layout.vertex_components[VARYING_SLOT_VAR0 + i] == 4);
   for (unsigned i = 12; i < 20; ++i)
      assert(layout.vertex_components[VARYING_SLOT_VAR0 + i] == 1);

   nir_shader *pull_vs = hk_nir_mesh_emulation_pull_vs(
      &agx_nir_options, &layout);
   assert(pull_vs != NULL);
   wide_producer_binary_size = compile_producer(b.shader);
   wide_pull_vs_binary_size = compile_pull_vs(pull_vs);
}

static void
test_contract_boundaries(void)
{
   /* All 32 vec4 user locations are the reported 128-component limit.  At
    * 93 vertices this also lands just below the 48-KiB output-memory cap. */
   nir_builder limit = triangle_builder(32, 0);
   limit.shader->info.mesh.max_vertices_out = 93;
   struct hk_mesh_emulation_layout layout;
   assert(hk_nir_lower_mesh_emulation(limit.shader, &layout));
   assert(layout.vertex_stride_B == 528);
   assert(layout.output_memory_B == 49120);
   nir_shader *pull_vs = hk_nir_mesh_emulation_pull_vs(
      &agx_nir_options, &layout);
   assert(pull_vs != NULL);
   limit_producer_binary_size = compile_producer(limit.shader);
   limit_pull_vs_binary_size = compile_pull_vs(pull_vs);

   nir_builder over_memory = triangle_builder(32, 0);
   over_memory.shader->info.mesh.max_vertices_out = 94;
   assert(!hk_nir_lower_mesh_emulation(over_memory.shader, NULL));
   ralloc_free(over_memory.shader);

   nir_builder shared_limit = triangle_builder(0, 0);
   shared_limit.shader->info.shared_size = 28672;
   assert(hk_nir_lower_mesh_emulation(shared_limit.shader, NULL));
   assert(shared_limit.shader->info.shared_size == 28688);
   ralloc_free(shared_limit.shader);

   nir_builder over_shared = triangle_builder(0, 0);
   over_shared.shader->info.shared_size = 28673;
   assert(!hk_nir_lower_mesh_emulation(over_shared.shader, NULL));
   ralloc_free(over_shared.shader);
}

static void
test_admission_identity_lifetime(void)
{
   nir_builder first = triangle_builder(1, 1);
   nir_builder same = triangle_builder(1, 1);
   nir_builder distinct = triangle_builder(1, 1);
   nir_builder unsupported = triangle_builder(0, 0);
   unsupported.shader->info.outputs_written |= VARYING_BIT_PSIZ;
   nir_store_per_vertex_output(
      &unsupported, nir_imm_float(&unsupported, 1.0f),
      nir_imm_int(&unsupported, 0), nir_imm_int(&unsupported, 0),
      .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});

   /* Move the contract's private shared base so the admitted programs and
    * their cache identity must differ while remaining otherwise supported. */
   distinct.shader->info.shared_size = 84;

   struct hk_mesh_emulation_admission *a =
      hk_mesh_emulation_admission_create(first.shader, &agx_nir_options);
   struct hk_mesh_emulation_admission *b =
      hk_mesh_emulation_admission_create(same.shader, &agx_nir_options);
   struct hk_mesh_emulation_admission *c =
      hk_mesh_emulation_admission_create(distinct.shader, &agx_nir_options);
   assert(a && b && c);
   assert(!hk_mesh_emulation_admission_create(unsupported.shader,
                                               &agx_nir_options));

   /* Admission clones and lowers; its caller-owned API NIR remains intact. */
   assert(first.shader->info.stage == MESA_SHADER_MESH);
   assert(same.shader->info.stage == MESA_SHADER_MESH);
   assert(distinct.shader->info.stage == MESA_SHADER_MESH);
   assert(unsupported.shader->info.stage == MESA_SHADER_MESH);
   assert(unsupported.shader->info.outputs_written & VARYING_BIT_PSIZ);

   assert(memcmp(hk_mesh_emulation_admission_identity(a),
                 hk_mesh_emulation_admission_identity(b),
                 BLAKE3_KEY_LEN) == 0);
   assert(memcmp(hk_mesh_emulation_admission_identity(a),
                 hk_mesh_emulation_admission_identity(c),
                 BLAKE3_KEY_LEN) != 0);
   assert(hk_mesh_emulation_admission_layout(a)->vertex_count == 24);
   assert(hk_mesh_emulation_admission_layout(c)->vertex_count == 88);

   nir_shader *producer = hk_mesh_emulation_admission_clone_producer(
      a, &agx_nir_options);
   nir_shader *pull_vs = hk_mesh_emulation_admission_clone_pull_vs(
      a, &agx_nir_options);
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

static nir_builder
task_builder(uint32_t x, uint32_t y, uint32_t z)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "G13 software task producer");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.task_payload_size = 256;
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_store_task_payload(&b, lane, nir_imul_imm(&b, lane, 4),
                          .align_mul = 4);
   nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, x, y, z));
   return b;
}

static void
test_task_producer(void)
{
   nir_builder b = task_builder(2, 1, 1);
   struct hk_task_emulation_layout layout;
   assert(hk_nir_lower_task_emulation(b.shader, &layout));
   nir_validate_shader(b.shader, "G13 software task producer lowered");
   assert(b.shader->info.stage == MESA_SHADER_COMPUTE);
   assert(b.shader->info.task_payload_size == 0);
   assert(layout.payload_size_B == 256);
   assert(layout.task_record_stride_B == 272);
   assert(layout.local_size[0] == 32 && layout.local_size[1] == 1 &&
          layout.local_size[2] == 1);

   unsigned launches = 0;
   unsigned payload_ops = 0;
   unsigned atomics = 0;
   unsigned barriers = 0;
   unsigned lane_ids = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            launches += intr->intrinsic ==
                        nir_intrinsic_launch_mesh_workgroups;
            payload_ops += intr->intrinsic == nir_intrinsic_load_task_payload ||
                           intr->intrinsic == nir_intrinsic_store_task_payload;
            atomics += intr->intrinsic == nir_intrinsic_global_atomic;
            barriers += intr->intrinsic == nir_intrinsic_barrier;
            lane_ids += intr->intrinsic ==
                        nir_intrinsic_load_local_invocation_index;
         }
      }
   }
   /* One static reservation/update sequence, guarded by lane zero, serves
    * the 32-invocation task workgroup. */
   assert(launches == 0 && payload_ops == 0);
   assert(atomics == 3);
   assert(barriers >= 1);
   assert(lane_ids >= 1);
   task_producer_binary_size = compile_producer(b.shader);

   /* A huge child grid takes the cap+1 failure sentinel.  Its generated map
    * loop has zero iterations on that arm and remains bounded at 4096 for
    * every admitted launch; lowering and AGX2 compile must terminate. */
   nir_builder first_overflow = task_builder(4097, 1, 1);
   assert(hk_nir_lower_task_emulation(first_overflow.shader, NULL));
   assert(compile_producer(first_overflow.shader) > 0);

   /* Two task workgroups emitting 3000 children each exercise the aggregate
    * overflow arm at execution time.  Lowering is independent of the parent
    * grid and must keep the same bounded sentinel/map loop; the command test
    * records its post-task overflow sanitizer before the indirect mesh kick. */
   nir_builder aggregate_overflow = task_builder(3000, 1, 1);
   assert(hk_nir_lower_task_emulation(aggregate_overflow.shader, NULL));
   assert(compile_producer(aggregate_overflow.shader) > 0);

   /* One static EmitMeshTasksEXT is the complete admitted contract.  Vulkan
    * validation supplies uniform control flow; reject NIR that could reserve
    * twice even when both launches are individually small. */
   nir_builder duplicate = task_builder(1, 1, 1);
   nir_launch_mesh_workgroups(&duplicate, nir_imm_ivec3(&duplicate, 1, 1, 1));
   assert(!hk_nir_lower_task_emulation(duplicate.shader, NULL));
   ralloc_free(duplicate.shader);
}

static void
test_task_mesh_interface(void)
{
   nir_builder b = triangle_builder(0, 0);
   b.shader->info.task_payload_size = 256;
   b.shader->info.outputs_written |= VARYING_BIT_VAR(0);
   nir_def *word = nir_load_task_payload(
      &b, 1, 32, nir_imm_int(&b, 0), .align_mul = 4);
   nir_def *value = nir_replicate(&b, word, 4);
   nir_store_per_vertex_output(
      &b, value, nir_load_local_invocation_index(&b), nir_imm_int(&b, 0),
      .write_mask = 0xf, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});

   struct hk_mesh_emulation_layout layout;
   assert(hk_nir_lower_mesh_emulation(b.shader, &layout));
   assert(layout.task_payload_size_B == 256);
   unsigned payload_ops = 0;
   unsigned global_loads = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            payload_ops += intr->intrinsic == nir_intrinsic_load_task_payload;
            global_loads += intr->intrinsic == nir_intrinsic_load_global;
         }
      }
   }
   assert(payload_ops == 0);
   assert(global_loads >= 4);
   task_mesh_binary_size = compile_producer(b.shader);
}

int
main(void)
{
   test_triangle();
   test_fail_closed();
   test_partial_components_and_types();
   test_pull_vs();
   test_title_inventory_width();
   test_contract_boundaries();
   test_admission_identity_lifetime();
   test_task_producer();
   test_task_mesh_interface();
   printf("G13 mesh NIR: producer PASS (%u B), pull VS PASS (%u B), "
          "56-component PASS (%u/%u B), limits PASS (%u/%u B), "
          "admission/cache PASS (%u/%u B), task PASS (%u/%u B), "
          "unsupported PASS\n",
          producer_binary_size, pull_vs_binary_size,
          wide_producer_binary_size, wide_pull_vs_binary_size,
          limit_producer_binary_size, limit_pull_vs_binary_size,
          admitted_producer_binary_size, admitted_pull_vs_binary_size,
          task_producer_binary_size, task_mesh_binary_size);
   return 0;
}
