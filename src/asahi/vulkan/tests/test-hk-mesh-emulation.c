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
   assert(offsetof(struct hk_mesh_emulation_params, vertex_buffer) == 0);
   assert(offsetof(struct hk_mesh_emulation_params, index_buffer) == 8);
   assert(sizeof(struct hk_mesh_emulation_params) == 16);

   unsigned geometry_params = 0;
   unsigned global_constants = 0;
   unsigned global_stores = 0;
   unsigned barriers = 0;
   unsigned mesh_intrinsics = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            geometry_params += intr->intrinsic ==
                               nir_intrinsic_load_geometry_param_buffer_poly;
            global_constants +=
               intr->intrinsic == nir_intrinsic_load_global_constant;
            global_stores += intr->intrinsic == nir_intrinsic_store_global;
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
   assert(global_constants == 2);
   assert(global_stores == 4);
   assert(barriers >= 3);
   assert(mesh_intrinsics == 0);

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
}

int
main(void)
{
   test_triangle();
   test_fail_closed();
   puts("G13 mesh emulation NIR contract: triangle PASS, unsupported PASS");
   return 0;
}
