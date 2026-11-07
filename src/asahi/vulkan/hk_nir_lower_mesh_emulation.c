/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_mesh_emulation.h"

#include <stddef.h>

#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"

#define HK_MESH_EMULATION_MAX_VERTICES   256
#define HK_MESH_EMULATION_MAX_PRIMITIVES 256

static struct hk_mesh_emulation_layout
mesh_layout(const nir_shader *shader)
{
   const unsigned base = ALIGN_POT(shader->info.shared_size, 64);
   const unsigned position = ALIGN_POT(base + 2 * sizeof(uint32_t), 16);
   const unsigned position_bytes =
      shader->info.mesh.max_vertices_out * 4 * sizeof(uint32_t);
   const unsigned indices = ALIGN_POT(position + position_bytes, 16);
   const unsigned index_bytes =
      shader->info.mesh.max_primitives_out * 3 * sizeof(uint32_t);

   return (struct hk_mesh_emulation_layout){
      .shared_base = base,
      .vertex_count = base,
      .primitive_count = base + sizeof(uint32_t),
      .position = position,
      .indices = indices,
      .shared_size = ALIGN_POT(indices + index_bytes, 64),
      .vertices_per_group = shader->info.mesh.max_vertices_out,
      .primitives_per_group = shader->info.mesh.max_primitives_out,
   };
}

static bool
validate_store(nir_intrinsic_instr *intr)
{
   if (!nir_src_is_const(intr->src[2]) || nir_src_as_uint(intr->src[2]) != 0)
      return false;

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (intr->intrinsic == nir_intrinsic_store_per_vertex_output) {
      return sem.location == VARYING_SLOT_POS && sem.num_slots == 1 &&
             nir_intrinsic_component(intr) == 0 &&
             nir_intrinsic_write_mask(intr) == 0xf &&
             intr->src[0].ssa->num_components == 4 &&
             intr->src[0].ssa->bit_size == 32;
   }

   return sem.location == VARYING_SLOT_PRIMITIVE_INDICES &&
          sem.num_slots == 1 &&
          nir_intrinsic_component(intr) == 0 &&
          nir_intrinsic_write_mask(intr) == 0x7 &&
          intr->src[0].ssa->num_components == 3 &&
          intr->src[0].ssa->bit_size == 32;
}

static bool
validate_shader(const nir_shader *shader,
                const struct hk_mesh_emulation_layout *layout)
{
   if (shader->info.stage != MESA_SHADER_MESH || !shader->info.io_lowered ||
       shader->info.mesh.primitive_type != MESA_PRIM_TRIANGLES ||
       shader->info.mesh.max_vertices_out == 0 ||
       shader->info.mesh.max_vertices_out > HK_MESH_EMULATION_MAX_VERTICES ||
       shader->info.mesh.max_primitives_out == 0 ||
       shader->info.mesh.max_primitives_out >
          HK_MESH_EMULATION_MAX_PRIMITIVES ||
       shader->info.workgroup_size_variable ||
       shader->info.task_payload_size != 0 ||
       shader->info.outputs_read != 0 ||
       shader->info.outputs_written !=
          (VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES) ||
       shader->info.per_primitive_outputs != VARYING_BIT_PRIMITIVE_INDICES ||
       layout->shared_size > 32 * 1024)
      return false;

   uint64_t threads = 1;
   for (unsigned i = 0; i < 3; ++i) {
      if (shader->info.workgroup_size[i] == 0)
         return false;
      threads *= shader->info.workgroup_size[i];
   }
   if (threads > 1024)
      return false;

   bool counts = false;
   bool position = false;
   bool indices = false;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_set_vertex_and_primitive_count:
               if (intr->src[0].ssa->bit_size != 32 ||
                   intr->src[1].ssa->bit_size != 32)
                  return false;
               counts = true;
               break;
            case nir_intrinsic_store_per_vertex_output:
            case nir_intrinsic_store_per_primitive_output:
               if (!validate_store(intr))
                  return false;
               position |= intr->intrinsic ==
                           nir_intrinsic_store_per_vertex_output;
               indices |= intr->intrinsic ==
                          nir_intrinsic_store_per_primitive_output;
               break;
            case nir_intrinsic_load_per_vertex_output:
            case nir_intrinsic_load_per_primitive_output:
            case nir_intrinsic_launch_mesh_workgroups:
            case nir_intrinsic_launch_mesh_workgroups_with_payload_deref:
            case nir_intrinsic_load_task_payload:
            case nir_intrinsic_store_task_payload:
               return false;
            default:
               break;
            }
         }
      }
   }

   return counts && position && indices;
}

static bool
lower_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct hk_mesh_emulation_layout *layout = data;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_store_per_vertex_output: {
      nir_def *offset = nir_iadd_imm(
         b, nir_imul_imm(b, intr->src[1].ssa, 4 * sizeof(uint32_t)),
         layout->position);
      nir_store_shared(b, intr->src[0].ssa, offset,
                       .write_mask = 0xf, .align_mul = 16);
      break;
   }
   case nir_intrinsic_store_per_primitive_output: {
      nir_def *offset = nir_iadd_imm(
         b, nir_imul_imm(b, intr->src[1].ssa, 3 * sizeof(uint32_t)),
         layout->indices);
      nir_store_shared(b, intr->src[0].ssa, offset,
                       .write_mask = 0x7, .align_mul = 4);
      break;
   }
   case nir_intrinsic_set_vertex_and_primitive_count:
      nir_push_if(
         b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_store_shared(
         b, nir_umin(b, intr->src[0].ssa,
                     nir_imm_int(b, layout->vertices_per_group)),
         nir_imm_int(b, layout->vertex_count),
         .write_mask = 1, .align_mul = 4);
      nir_store_shared(
         b, nir_umin(b, intr->src[1].ssa,
                     nir_imm_int(b, layout->primitives_per_group)),
         nir_imm_int(b, layout->primitive_count),
         .write_mask = 1, .align_mul = 4);
      nir_pop_if(b, NULL);
      break;
   default:
      return false;
   }

   nir_instr_remove(&intr->instr);
   return true;
}

static nir_def *
linear_workgroup_id(nir_builder *b)
{
   nir_def *id = nir_load_workgroup_id(b);
   nir_def *count = nir_load_num_workgroups(b);
   nir_def *yz = nir_iadd(
      b, nir_channel(b, id, 1),
      nir_imul(b, nir_channel(b, count, 1), nir_channel(b, id, 2)));

   return nir_iadd(b, nir_channel(b, id, 0),
                   nir_imul(b, nir_channel(b, count, 0), yz));
}

static nir_def *
load_param(nir_builder *b, nir_def *params, unsigned offset)
{
   return nir_load_global_constant(b, 1, 64, nir_iadd_imm(b, params, offset),
                                   .align_mul = 8,
                                   .access = ACCESS_CAN_SPECULATE);
}

static void
initialize_scratch(nir_builder *b,
                   const struct hk_mesh_emulation_layout *layout,
                   unsigned threads)
{
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_store_shared(b, nir_imm_int(b, 0),
                    nir_imm_int(b, layout->vertex_count),
                    .write_mask = 1, .align_mul = 4);
   nir_store_shared(b, nir_imm_int(b, 0),
                    nir_imm_int(b, layout->primitive_count),
                    .write_mask = 1, .align_mul = 4);
   nir_pop_if(b, NULL);

   for (unsigned base = 0; base < layout->vertices_per_group;
        base += threads) {
      nir_def *vertex = nir_iadd_imm(b, lane, base);
      nir_push_if(b, nir_ult_imm(b, vertex, layout->vertices_per_group));
      nir_store_shared(
         b, nir_imm_zero(b, 4, 32),
         nir_iadd_imm(b, nir_imul_imm(b, vertex, 16), layout->position),
         .write_mask = 0xf, .align_mul = 16);
      nir_pop_if(b, NULL);
   }

   for (unsigned base = 0; base < layout->primitives_per_group;
        base += threads) {
      nir_def *primitive = nir_iadd_imm(b, lane, base);
      nir_push_if(b,
                  nir_ult_imm(b, primitive, layout->primitives_per_group));
      nir_store_shared(
         b, nir_imm_zero(b, 3, 32),
         nir_iadd_imm(b, nir_imul_imm(b, primitive, 12), layout->indices),
         .write_mask = 0x7, .align_mul = 4);
      nir_pop_if(b, NULL);
   }

   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

static void
export_scratch(nir_builder *b,
               const struct hk_mesh_emulation_layout *layout,
               unsigned threads)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   nir_def *params = nir_load_geometry_param_buffer_poly(b);
   nir_def *vertex_buffer =
      load_param(b, params, offsetof(struct hk_mesh_emulation_params,
                                     vertex_buffer));
   nir_def *index_buffer =
      load_param(b, params,
                 offsetof(struct hk_mesh_emulation_params, index_buffer));
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *group = linear_workgroup_id(b);
   nir_def *vertex_base =
      nir_imul_imm(b, group, layout->vertices_per_group);
   nir_def *primitive_base =
      nir_imul_imm(b, group, layout->primitives_per_group);
   nir_def *primitive_count = nir_load_shared(
      b, 1, 32, nir_imm_int(b, layout->primitive_count), .align_mul = 4);

   for (unsigned base = 0; base < layout->vertices_per_group;
        base += threads) {
      nir_def *vertex = nir_iadd_imm(b, lane, base);
      nir_push_if(b, nir_ult_imm(b, vertex, layout->vertices_per_group));
      nir_def *value = nir_load_shared(
         b, 4, 32,
         nir_iadd_imm(b, nir_imul_imm(b, vertex, 16), layout->position),
         .align_mul = 16);
      nir_def *address = nir_iadd(
         b, vertex_buffer,
         nir_u2u64(b, nir_imul_imm(
                         b, nir_iadd(b, vertex_base, vertex), 16)));
      nir_store_global(b, value, address,
                       .write_mask = 0xf, .align_mul = 16);
      nir_pop_if(b, NULL);
   }

   for (unsigned base = 0; base < layout->primitives_per_group;
        base += threads) {
      nir_def *primitive = nir_iadd_imm(b, lane, base);
      nir_push_if(b,
                  nir_ult_imm(b, primitive, layout->primitives_per_group));
      nir_def *indices = nir_load_shared(
         b, 3, 32,
         nir_iadd_imm(b, nir_imul_imm(b, primitive, 12), layout->indices),
         .align_mul = 4);
      nir_def *active = nir_ult(b, primitive, primitive_count);
      nir_def *adjusted = nir_iadd(b, indices, vertex_base);
      adjusted = nir_bcsel(b, active, adjusted,
                           nir_replicate(b, vertex_base, 3));
      nir_def *record = nir_iadd(b, primitive_base, primitive);
      for (unsigned c = 0; c < 3; ++c) {
         nir_def *word =
            nir_iadd_imm(b, nir_imul_imm(b, record, 3), c);
         nir_def *address =
            nir_ulea_agx(b, index_buffer, word, nir_imm_int(b, 2));
         nir_store_global(b, nir_channel(b, adjusted, c), address,
                          .write_mask = 1, .align_mul = 4);
      }
      nir_pop_if(b, NULL);
   }

   nir_barrier(b, .execution_scope = SCOPE_NONE,
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_RELEASE,
               .memory_modes = nir_var_mem_global);
}

bool
hk_nir_lower_mesh_emulation(
   nir_shader *shader, struct hk_mesh_emulation_layout *layout_out)
{
   if (shader->info.stage != MESA_SHADER_MESH ||
       shader->info.shared_size > 32 * 1024)
      return false;

   struct hk_mesh_emulation_layout layout = mesh_layout(shader);
   if (!validate_shader(shader, &layout))
      return false;

   const unsigned threads = shader->info.workgroup_size[0] *
                            shader->info.workgroup_size[1] *
                            shader->info.workgroup_size[2];
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_builder prologue = nir_builder_at(nir_before_impl(impl));
   initialize_scratch(&prologue, &layout, threads);

   bool progress = nir_shader_intrinsics_pass(
      shader, lower_output, nir_metadata_none, &layout);
   if (!progress)
      return false;

   nir_builder epilogue = nir_builder_at(nir_after_cf_list(&impl->body));
   export_scratch(&epilogue, &layout, threads);

   shader->info.stage = MESA_SHADER_COMPUTE;
   shader->info.shared_size = layout.shared_size;
   shader->info.outputs_written = 0;
   shader->info.per_primitive_outputs = 0;
   shader->info.outputs_read = 0;
   nir_shader_gather_info(shader, impl);
   nir_progress(true, impl, nir_metadata_none);

   if (layout_out)
      *layout_out = layout;

   return true;
}
