/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_lower_mesh_emulation.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include "compiler/nir/nir_builder.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "util/u_atomic.h"
#include "util/u_math.h"

#define HK_MESH_EMULATION_MAX_VERTICES   256
#define HK_MESH_EMULATION_MAX_PRIMITIVES 256
#define HK_MESH_EMULATION_ADMISSION_VERSION HK_MESH_EMULATION_ABI_VERSION

struct hk_mesh_emulation_admission {
   uint32_t refcount;
   uint8_t identity[BLAKE3_KEY_LEN];
   struct hk_mesh_emulation_layout layout;
   void *producer;
   size_t producer_size;
   void *pull_vs;
   size_t pull_vs_size;
};

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
            if (instr->type == nir_instr_type_jump &&
                nir_instr_as_jump(instr)->type == nir_jump_return) {
               /* The appended export contains a workgroup barrier.  Until
                * the pass lowers returns into structured control flow,
                * admitting a returning lane could make it non-uniform. */
               return false;
            }

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
linear_workgroup_id(nir_builder *b, nir_def *count)
{
   nir_def *id = nir_load_workgroup_id(b);
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

static nir_def *
load_param_u32(nir_builder *b, nir_def *params, unsigned offset,
               unsigned components)
{
   return nir_load_global_constant(b, components, 32,
                                   nir_iadd_imm(b, params, offset),
                                   .align_mul = 4,
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
   nir_def *vertex_records =
      load_param(b, params, offsetof(struct hk_mesh_emulation_params,
                                     vertex_records));
   nir_def *index_buffer =
      load_param(b, params,
                 offsetof(struct hk_mesh_emulation_params, index_buffer));
   nir_def *group_records =
      load_param(b, params,
                 offsetof(struct hk_mesh_emulation_params, group_records));
   nir_def *draw_params =
      load_param(b, params,
                 offsetof(struct hk_mesh_emulation_params, draw_params));
   nir_def *mesh_grid =
      load_param_u32(b, params,
                     offsetof(struct hk_mesh_emulation_params, mesh_grid), 3);
   nir_def *abi_version =
      load_param_u32(b, params,
                     offsetof(struct hk_mesh_emulation_params, abi_version),
                     1);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *group = linear_workgroup_id(b, mesh_grid);
   nir_def *vertex_base =
      nir_imul_imm(b, group, layout->vertices_per_group);
   nir_def *primitive_base =
      nir_imul_imm(b, group, layout->primitives_per_group);
   nir_def *primitive_count = nir_load_shared(
      b, 1, 32, nir_imm_int(b, layout->primitive_count), .align_mul = 4);
   nir_def *vertex_count = nir_load_shared(
      b, 1, 32, nir_imm_int(b, layout->vertex_count), .align_mul = 4);
   nir_def *valid =
      nir_ieq_imm(b, abi_version, HK_MESH_EMULATION_ABI_VERSION);
   /* AGX2 ULEA encodes shifts through four. Address 32-byte records as
    * 16-byte records at an even index, keeping this as one grounded ULEA. */
   nir_def *group_record = nir_ulea_agx(
      b, group_records, nir_imul_imm(b, group, 2), nir_imm_int(b, 4));
   /* VkDrawIndexedIndirectCommand is five dwords. Address one GPU-owned draw
    * record per mesh workgroup as (group * 5) << 2. */
   nir_def *draw_record = nir_ulea_agx(
      b, draw_params, nir_imul_imm(b, group, 5), nir_imm_int(b, 2));

   nir_push_if(b, valid);

   for (unsigned base = 0; base < layout->vertices_per_group;
        base += threads) {
      nir_def *vertex = nir_iadd_imm(b, lane, base);
      nir_push_if(b, nir_ult_imm(b, vertex, layout->vertices_per_group));
      nir_def *value = nir_load_shared(
         b, 4, 32,
         nir_iadd_imm(b, nir_imul_imm(b, vertex, 16), layout->position),
         .align_mul = 16);
      nir_def *address =
         nir_ulea_agx(b, vertex_records, nir_iadd(b, vertex_base, vertex),
                      nir_imm_int(b, 4));
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

   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_store_global(
      b, nir_vec4(b, vertex_count, primitive_count, vertex_base,
                  primitive_base),
      group_record, .write_mask = 0xf, .align_mul = 16);

   nir_def *index_count = nir_imul_imm(b, primitive_count, 3);
   nir_def *first_index = nir_imul_imm(b, primitive_base, 3);
   nir_store_global(
      b, nir_vec4(b, index_count, nir_imm_int(b, 1), first_index,
                  nir_imm_int(b, 0)),
      draw_record, .write_mask = 0xf, .align_mul = 4);
   nir_pop_if(b, NULL);

   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_store_global(b, nir_imm_zero(b, 4, 32), group_record,
                    .write_mask = 0xf, .align_mul = 16);
   nir_store_global(b, nir_imm_zero(b, 4, 32), draw_record,
                    .write_mask = 0xf, .align_mul = 4);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);

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

nir_shader *
hk_nir_mesh_emulation_pull_vs(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, options, "G13 software mesh pull VS");
   nir_def *params = nir_load_geometry_param_buffer_poly(&b);
   nir_def *vertex_records =
      load_param(&b, params,
                 offsetof(struct hk_mesh_emulation_params, vertex_records));
   nir_def *vertex = nir_load_vertex_id(&b);
   nir_def *address =
      nir_ulea_agx(&b, vertex_records, vertex, nir_imm_int(&b, 4));
   nir_def *position = nir_load_global_constant(
      &b, 4, 32, address, .align_mul = 16,
      .access = ACCESS_CAN_SPECULATE);

   nir_store_output(
      &b, position, nir_imm_int(&b, 0), .write_mask = 0xf,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   b.shader->info.outputs_written = VARYING_BIT_POS;
   b.shader->info.io_lowered = true;
   nir_shader_gather_info(b.shader, b.impl);

   return b.shader;
}

static bool
serialize_nir(const nir_shader *nir, void **data_out, size_t *size_out)
{
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, false);
   if (blob.out_of_memory) {
      blob_finish(&blob);
      return false;
   }

   blob_finish_get_buffer(&blob, data_out, size_out);
   return true;
}

static nir_shader *
deserialize_nir(const void *data, size_t size,
                const nir_shader_compiler_options *options)
{
   struct blob_reader reader;
   blob_reader_init(&reader, data, size);
   nir_shader *nir = nir_deserialize(NULL, options, &reader);
   if (!nir || reader.overrun || reader.current != reader.end) {
      ralloc_free(nir);
      return NULL;
   }

   return nir;
}

struct hk_mesh_emulation_admission *
hk_mesh_emulation_admission_create(
   const nir_shader *mesh, const nir_shader_compiler_options *options)
{
   nir_shader *producer = nir_shader_clone(NULL, mesh);
   if (!producer)
      return NULL;

   struct hk_mesh_emulation_layout layout;
   if (!hk_nir_lower_mesh_emulation(producer, &layout)) {
      ralloc_free(producer);
      return NULL;
   }

   nir_shader *pull_vs = hk_nir_mesh_emulation_pull_vs(options);
   if (!pull_vs) {
      ralloc_free(producer);
      return NULL;
   }

   struct hk_mesh_emulation_admission *admission =
      calloc(1, sizeof(*admission));
   if (!admission ||
       !serialize_nir(producer, &admission->producer,
                      &admission->producer_size) ||
       !serialize_nir(pull_vs, &admission->pull_vs,
                      &admission->pull_vs_size)) {
      if (admission) {
         free(admission->producer);
         free(admission->pull_vs);
      }
      free(admission);
      ralloc_free(producer);
      ralloc_free(pull_vs);
      return NULL;
   }

   admission->refcount = 1;
   admission->layout = layout;

   const uint32_t version = HK_MESH_EMULATION_ADMISSION_VERSION;
   const uint32_t target = HK_MESH_EMULATION_TARGET_G13;
   const uint64_t producer_size = admission->producer_size;
   const uint64_t pull_vs_size = admission->pull_vs_size;
   blake3_hasher hasher;
   _mesa_blake3_init(&hasher);
   _mesa_blake3_update(&hasher, &version, sizeof(version));
   _mesa_blake3_update(&hasher, &target, sizeof(target));
   _mesa_blake3_update(&hasher, &producer_size, sizeof(producer_size));
   _mesa_blake3_update(&hasher, admission->producer,
                       admission->producer_size);
   _mesa_blake3_update(&hasher, &pull_vs_size, sizeof(pull_vs_size));
   _mesa_blake3_update(&hasher, admission->pull_vs,
                       admission->pull_vs_size);
   _mesa_blake3_final(&hasher, admission->identity);

   ralloc_free(producer);
   ralloc_free(pull_vs);
   return admission;
}

struct hk_mesh_emulation_admission *
hk_mesh_emulation_admission_ref(struct hk_mesh_emulation_admission *admission)
{
   assert(admission && p_atomic_read(&admission->refcount) > 0);
   p_atomic_inc(&admission->refcount);
   return admission;
}

bool
hk_mesh_emulation_admission_unref(
   struct hk_mesh_emulation_admission *admission)
{
   if (!admission)
      return false;

   assert(p_atomic_read(&admission->refcount) > 0);
   if (!p_atomic_dec_zero(&admission->refcount))
      return false;

   free(admission->producer);
   free(admission->pull_vs);
   free(admission);
   return true;
}

const uint8_t *
hk_mesh_emulation_admission_identity(
   const struct hk_mesh_emulation_admission *admission)
{
   return admission->identity;
}

const struct hk_mesh_emulation_layout *
hk_mesh_emulation_admission_layout(
   const struct hk_mesh_emulation_admission *admission)
{
   return &admission->layout;
}

nir_shader *
hk_mesh_emulation_admission_clone_producer(
   const struct hk_mesh_emulation_admission *admission,
   const nir_shader_compiler_options *options)
{
   return deserialize_nir(admission->producer, admission->producer_size,
                          options);
}

nir_shader *
hk_mesh_emulation_admission_clone_pull_vs(
   const struct hk_mesh_emulation_admission *admission,
   const nir_shader_compiler_options *options)
{
   return deserialize_nir(admission->pull_vs, admission->pull_vs_size,
                          options);
}
