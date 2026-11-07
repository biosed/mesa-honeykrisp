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
/* The title inventory reaches 37,376 bytes.  Keep the software contract on a
 * documented 16-KiB boundary above that rather than inheriting the extension
 * minimum, which is too small for the complete known shader set. */
#define HK_MESH_EMULATION_MAX_OUTPUT_MEMORY (48 * 1024)
#define HK_MESH_EMULATION_ADMISSION_VERSION HK_MESH_EMULATION_ABI_VERSION
#define HK_MESH_EMULATION_USER_OUTPUTS \
   BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32)

struct hk_mesh_emulation_admission {
   uint32_t refcount;
   uint8_t identity[BLAKE3_KEY_LEN];
   struct hk_mesh_emulation_layout layout;
   void *producer;
   size_t producer_size;
   void *pull_vs;
   size_t pull_vs_size;
};

static_assert(sizeof(struct hk_mesh_emulation_params) ==
                 HK_MESH_EMULATION_PARAMS_SIZE,
              "software mesh parameter ABI size");

static bool
task_payload_access_supported(const nir_intrinsic_instr *intr, bool store)
{
   const unsigned bit_size = store ? intr->src[0].ssa->bit_size
                                   : intr->def.bit_size;
   const unsigned bytes = bit_size / 8;
   if (!util_is_power_of_two_or_zero(bytes) || bytes == 0 || bytes > 8 ||
       nir_intrinsic_align_mul(intr) < bytes)
      return false;

   return ((HK_MESH_EMULATION_TASK_RECORD_HEADER_SIZE +
            nir_intrinsic_base(intr) + nir_intrinsic_align_offset(intr)) &
           (bytes - 1)) == 0;
}

static bool
record_output(struct hk_mesh_emulation_layout *layout,
              nir_intrinsic_instr *intr, bool primitive)
{
   if (!nir_src_is_const(intr->src[2]) ||
       intr->src[0].ssa->bit_size != 32)
      return false;

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   const unsigned slot = sem.location + nir_src_as_uint(intr->src[2]);
   const unsigned component = nir_intrinsic_component(intr);
   const unsigned components = component +
      util_last_bit(nir_intrinsic_write_mask(intr));

   if (!primitive && slot == VARYING_SLOT_POS) {
      return sem.num_slots == 1 && component == 0 && components == 4 &&
             nir_intrinsic_write_mask(intr) == 0xf &&
             intr->src[0].ssa->num_components == 4 &&
             nir_intrinsic_src_type(intr) == nir_type_float32;
   }

   if (primitive && slot == VARYING_SLOT_PRIMITIVE_INDICES) {
      return sem.num_slots == 1 && component == 0 && components == 3 &&
             nir_intrinsic_write_mask(intr) == 0x7 &&
             intr->src[0].ssa->num_components == 3 &&
             nir_intrinsic_src_type(intr) == nir_type_uint32;
   }

   if (slot < VARYING_SLOT_VAR0 || slot > VARYING_SLOT_VAR31 ||
       components == 0 || components > 4)
      return false;

   const nir_alu_type type = nir_intrinsic_src_type(intr);
   const nir_alu_type base = nir_alu_type_get_base_type(type);
   if (nir_alu_type_get_type_size(type) != 32 ||
       (base != nir_type_float && base != nir_type_int &&
        base != nir_type_uint))
      return false;

   uint8_t *count = primitive ? layout->primitive_components
                              : layout->vertex_components;
   uint8_t *other = primitive ? layout->vertex_components
                              : layout->primitive_components;
   if (other[slot] ||
       (layout->output_type[slot] && layout->output_type[slot] != type))
      return false;

   count[slot] = MAX2(count[slot], components);
   layout->output_type[slot] = type;
   if (primitive)
      layout->primitive_outputs |= BITFIELD64_BIT(slot);
   else
      layout->vertex_outputs |= BITFIELD64_BIT(slot);

   return true;
}

static bool
validate_shader(const nir_shader *shader,
                struct hk_mesh_emulation_layout *layout)
{
   memset(layout, 0, sizeof(*layout));
   memset(layout->vertex_offset_B, 0xff, sizeof(layout->vertex_offset_B));
   memset(layout->primitive_offset_B, 0xff,
          sizeof(layout->primitive_offset_B));
   memset(layout->expanded_offset_B, 0xff,
          sizeof(layout->expanded_offset_B));

   if (shader->info.stage != MESA_SHADER_MESH || !shader->info.io_lowered ||
       shader->info.mesh.primitive_type != MESA_PRIM_TRIANGLES ||
       shader->info.mesh.max_vertices_out == 0 ||
       shader->info.mesh.max_vertices_out > HK_MESH_EMULATION_MAX_VERTICES ||
       shader->info.mesh.max_primitives_out == 0 ||
       shader->info.mesh.max_primitives_out >
          HK_MESH_EMULATION_MAX_PRIMITIVES ||
       shader->info.workgroup_size_variable ||
       shader->info.task_payload_size >
          HK_MESH_EMULATION_MAX_TASK_PAYLOAD ||
       shader->info.outputs_read != 0 ||
       (shader->info.outputs_written &
        ~(VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES |
          HK_MESH_EMULATION_USER_OUTPUTS)) ||
       (shader->info.per_primitive_outputs &
        ~(VARYING_BIT_PRIMITIVE_INDICES |
          HK_MESH_EMULATION_USER_OUTPUTS)) ||
       shader->info.shared_size > 28672)
      return false;

   const uint32_t max_workgroup_size[3] = {
      HK_MESH_EMULATION_MAX_WORKGROUP_X,
      HK_MESH_EMULATION_MAX_WORKGROUP_Y,
      HK_MESH_EMULATION_MAX_WORKGROUP_Z,
   };
   uint64_t threads = 1;
   for (unsigned i = 0; i < 3; ++i) {
      if (shader->info.workgroup_size[i] == 0 ||
          shader->info.workgroup_size[i] > max_workgroup_size[i])
         return false;
      threads *= shader->info.workgroup_size[i];
   }
   if (threads > HK_MESH_EMULATION_MAX_WORKGROUP_INVOCATIONS)
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
               if (!record_output(
                      layout, intr,
                      intr->intrinsic ==
                         nir_intrinsic_store_per_primitive_output))
                  return false;
               position |= intr->intrinsic ==
                              nir_intrinsic_store_per_vertex_output &&
                           nir_intrinsic_io_semantics(intr).location +
                                 nir_src_as_uint(intr->src[2]) ==
                              VARYING_SLOT_POS;
               indices |= intr->intrinsic ==
                             nir_intrinsic_store_per_primitive_output &&
                          nir_intrinsic_io_semantics(intr).location +
                                nir_src_as_uint(intr->src[2]) ==
                             VARYING_SLOT_PRIMITIVE_INDICES;
               break;
            case nir_intrinsic_load_per_vertex_output:
            case nir_intrinsic_load_per_primitive_output:
            case nir_intrinsic_launch_mesh_workgroups:
            case nir_intrinsic_launch_mesh_workgroups_with_payload_deref:
            case nir_intrinsic_store_task_payload:
               return false;
            case nir_intrinsic_load_task_payload:
               if (!shader->info.task_payload_size)
                  return false;
               if (!task_payload_access_supported(intr, false))
                  return false;
               break;
            default:
               break;
            }
         }
      }
   }

   if (!counts || !position || !indices)
      return false;

   layout->vertices_per_group = shader->info.mesh.max_vertices_out;
   layout->primitives_per_group = shader->info.mesh.max_primitives_out;
   layout->vertex_count = ALIGN_POT(shader->info.shared_size, 8);
   layout->primitive_count = layout->vertex_count + sizeof(uint32_t);
   layout->shared_size = ALIGN_POT(layout->primitive_count +
                                   sizeof(uint32_t), 16);
   layout->task_payload_size_B = shader->info.task_payload_size;
   if (layout->shared_size > 32 * 1024)
      return false;

   layout->position_offset_B = 0;
   layout->index_offset_B = 0;
   layout->vertex_stride_B = 16;
   layout->primitive_stride_B = 16;
   layout->expanded_stride_B = 16;
   layout->vertex_outputs |= VARYING_BIT_POS;
   layout->primitive_outputs |= VARYING_BIT_PRIMITIVE_INDICES;

   /* Vulkan's maxMeshOutputComponents counts user-defined components;
    * built-ins such as Position are governed separately. */
   unsigned output_components = 0;
   for (unsigned slot = VARYING_SLOT_VAR0;
        slot <= VARYING_SLOT_VAR31; ++slot) {
      const unsigned vertex = layout->vertex_components[slot];
      const unsigned primitive = layout->primitive_components[slot];
      const unsigned components = MAX2(vertex, primitive);
      if (!components)
         continue;

      output_components += components;
      if (output_components > 128)
         return false;

      if (vertex) {
         layout->vertex_offset_B[slot] = layout->vertex_stride_B;
         layout->vertex_stride_B += vertex * sizeof(uint32_t);
      } else {
         layout->primitive_offset_B[slot] = layout->primitive_stride_B;
         layout->primitive_stride_B += primitive * sizeof(uint32_t);
      }
      layout->expanded_offset_B[slot] = layout->expanded_stride_B;
      layout->expanded_stride_B += components * sizeof(uint32_t);
   }

   layout->vertex_stride_B = ALIGN_POT(layout->vertex_stride_B, 16);
   layout->primitive_stride_B = ALIGN_POT(layout->primitive_stride_B, 16);
   layout->expanded_stride_B = ALIGN_POT(layout->expanded_stride_B, 16);
   const uint64_t output_memory =
      (uint64_t)layout->vertices_per_group * layout->vertex_stride_B +
      (uint64_t)layout->primitives_per_group * layout->primitive_stride_B;
   if (output_memory > HK_MESH_EMULATION_MAX_OUTPUT_MEMORY)
      return false;
   layout->output_memory_B = output_memory;

   return true;
}

static nir_def *linear_workgroup_id(nir_builder *b, nir_def *count);
static nir_def *load_param(nir_builder *b, nir_def *params, unsigned offset);
static nir_def *load_param_u32(nir_builder *b, nir_def *params,
                               unsigned offset, unsigned components);
static nir_def *record_address(nir_builder *b, nir_def *base,
                               nir_def *index, unsigned stride_B);

static nir_def *
task_record_element_address(nir_builder *b, nir_def *records,
                            nir_def *task_index, unsigned stride_B)
{
   assert((stride_B & 3) == 0);
   return nir_ulea_agx(
      b, records, nir_imul_imm(b, task_index, stride_B / 4),
      nir_imm_int(b, 2));
}

static nir_def *
task_payload_element_address(nir_builder *b, nir_def *records,
                             nir_def *task_index, unsigned stride_B,
                             nir_def *dynamic_offset_B,
                             unsigned constant_offset_B,
                             unsigned bit_size)
{
   const unsigned bytes = bit_size / 8;
   if (!util_is_power_of_two_or_zero(bytes) || bytes == 0 || bytes > 8)
      return NULL;
   const unsigned shift = util_logbase2(bytes);
   nir_def *offset_B = nir_iadd(
      b, nir_imul_imm(b, task_index, stride_B),
      nir_iadd_imm(b, dynamic_offset_B, constant_offset_B));
   return nir_ulea_agx(b, records, nir_ushr_imm(b, offset_B, shift),
                       nir_imm_int(b, shift));
}

/* Recover the API mesh identity selected by its parent task.  The hardware
 * launch is a compact one-dimensional indirect dispatch; this map restores
 * the task index and the three-dimensional child WorkGroupID. */
static bool
lower_mesh_task_interface(nir_builder *b, nir_intrinsic_instr *intr,
                          void *data)
{
   const struct hk_mesh_emulation_layout *layout = data;
   if (!layout->task_payload_size_B)
      return false;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_task_payload:
   case nir_intrinsic_barrier:
      break;
   default:
      return false;
   }

   if (intr->intrinsic == nir_intrinsic_barrier &&
       !(nir_intrinsic_memory_modes(intr) & nir_var_mem_task_payload))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (intr->intrinsic == nir_intrinsic_barrier) {
      nir_intrinsic_set_memory_modes(
         intr, (nir_intrinsic_memory_modes(intr) &
                ~nir_var_mem_task_payload) |
                  nir_var_mem_global);
      return true;
   }

   nir_def *params = nir_load_geometry_param_buffer_poly(b);
   nir_def *mesh_grid = load_param_u32(
      b, params, offsetof(struct hk_mesh_emulation_params, mesh_grid), 3);
   nir_def *physical_group = linear_workgroup_id(b, mesh_grid);
   nir_def *map_base = load_param(
      b, params, offsetof(struct hk_mesh_emulation_params, group_to_task));
   nir_def *map = record_address(
      b, map_base, physical_group,
      sizeof(struct hk_mesh_emulation_group_task));
   /* The task dispatch wrote this mapping before the pre-gfx barrier.  It is
    * mutable cross-dispatch memory, not a constant/speculatable table. */
   nir_def *mapping = nir_load_global(
      b, 2, 32, map, .align_mul = 8);
   nir_def *task_index = nir_channel(b, mapping, 0);
   nir_def *local_group = nir_channel(b, mapping, 1);
   nir_def *task_records = load_param(
      b, params, offsetof(struct hk_mesh_emulation_params, task_records));
   const unsigned task_stride = ALIGN_POT(
      HK_MESH_EMULATION_TASK_RECORD_HEADER_SIZE +
         layout->task_payload_size_B,
      16);
   nir_def *task_record = task_record_element_address(
      b, task_records, task_index, task_stride);
   nir_def *child_grid = nir_load_global(
      b, 3, 32, task_record, .align_mul = 16);

   if (intr->intrinsic == nir_intrinsic_load_num_workgroups) {
      nir_def_replace(&intr->def,
                      nir_u2uN(b, child_grid, intr->def.bit_size));
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_workgroup_id) {
      nir_def *x = nir_channel(b, child_grid, 0);
      nir_def *y = nir_channel(b, child_grid, 1);
      nir_def *xy = nir_imul(b, x, y);
      nir_def *id = nir_vec3(
         b, nir_umod(b, local_group, x),
         nir_umod(b, nir_udiv(b, local_group, x), y),
         nir_udiv(b, local_group, xy));
      nir_def_replace(&intr->def, nir_u2uN(b, id, intr->def.bit_size));
      return true;
   }

   assert(intr->intrinsic == nir_intrinsic_load_task_payload);
   nir_def *address = task_payload_element_address(
      b, task_records, task_index, task_stride, intr->src[0].ssa,
      HK_MESH_EMULATION_TASK_RECORD_HEADER_SIZE + nir_intrinsic_base(intr),
      intr->def.bit_size);
   if (!address)
      return false;
   nir_def_replace(
      &intr->def,
      nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                      address, .align_mul = nir_intrinsic_align_mul(intr),
                      .align_offset = nir_intrinsic_align_offset(intr),
                      .access = nir_intrinsic_access(intr)));
   return true;
}

static nir_def *
mesh_task_global_id(nir_builder *b, unsigned bit_size)
{
   nir_def *group = nir_u2uN(b, nir_load_workgroup_id(b), bit_size);
   nir_def *local = nir_u2uN(b, nir_load_local_invocation_id(b), bit_size);
   nir_def *size = nir_imm_ivec3(
      b, b->shader->info.workgroup_size[0],
      b->shader->info.workgroup_size[1],
      b->shader->info.workgroup_size[2]);
   return nir_iadd(b, nir_imul(b, group, nir_u2uN(b, size, bit_size)), local);
}

/* Derived compute IDs must be expanded while WorkGroupId still denotes the
 * Vulkan mesh identity.  If left until the ordinary compute preprocessing
 * pass, that pass would derive them from the compact physical 1-D dispatch
 * introduced by task lowering rather than the reconstructed task child grid. */
static bool
lower_mesh_task_derived_ids(nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   const unsigned bit_size = intr->def.bit_size;
   nir_def *value;

   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global_invocation_id:
      value = mesh_task_global_id(b, bit_size);
      break;
   case nir_intrinsic_load_global_invocation_index: {
      nir_def *global = mesh_task_global_id(b, bit_size);
      nir_def *groups = nir_u2uN(b, nir_load_num_workgroups(b), bit_size);
      nir_def *size = nir_imm_ivec3(
         b, b->shader->info.workgroup_size[0],
         b->shader->info.workgroup_size[1],
         b->shader->info.workgroup_size[2]);
      nir_def *global_size = nir_imul(b, groups,
                                      nir_u2uN(b, size, bit_size));
      value = nir_imul(b, nir_channel(b, global, 2),
                       nir_channel(b, global_size, 1));
      value = nir_iadd(b, nir_channel(b, global, 1), value);
      value = nir_imul(b, nir_channel(b, global_size, 0), value);
      value = nir_iadd(b, nir_channel(b, global, 0), value);
      break;
   }
   case nir_intrinsic_load_workgroup_index: {
      nir_def *group = nir_u2uN(b, nir_load_workgroup_id(b), bit_size);
      nir_def *groups = nir_u2uN(b, nir_load_num_workgroups(b), bit_size);
      value = nir_imul(b, nir_channel(b, group, 2),
                       nir_channel(b, groups, 1));
      value = nir_iadd(b, nir_channel(b, group, 1), value);
      value = nir_imul(b, nir_channel(b, groups, 0), value);
      value = nir_iadd(b, nir_channel(b, group, 0), value);
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&intr->def, value);
   return true;
}

/* Every command-owned record region is bounded to UINT32_MAX bytes before
 * submission.  Keep the shader address arithmetic in that same checked
 * domain and use AGX's grounded base + u32 offset form, including for strides
 * such as 48 bytes which are not representable by ULEA's scale alone. */
static nir_def *
record_address(nir_builder *b, nir_def *base, nir_def *index,
               unsigned stride_B)
{
   return nir_ulea_agx(b, base, nir_imul_imm(b, index, stride_B),
                       nir_imm_int(b, 0));
}

static bool
lower_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct hk_mesh_emulation_layout *layout = data;
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *params = nir_load_geometry_param_buffer_poly(b);
   nir_def *mesh_grid =
      load_param_u32(b, params,
                     offsetof(struct hk_mesh_emulation_params, mesh_grid), 3);
   nir_def *group = linear_workgroup_id(b, mesh_grid);

   switch (intr->intrinsic) {
   case nir_intrinsic_store_per_vertex_output: {
      const unsigned slot = nir_intrinsic_io_semantics(intr).location +
                            nir_src_as_uint(intr->src[2]);
      const unsigned offset = slot == VARYING_SLOT_POS
         ? layout->position_offset_B : layout->vertex_offset_B[slot];
      nir_def *records =
         load_param(b, params,
                    offsetof(struct hk_mesh_emulation_params, vertex_records));
      nir_push_if(
         b, nir_ult_imm(b, intr->src[1].ssa, layout->vertices_per_group));
      nir_def *index = nir_iadd(
         b, nir_imul_imm(b, group, layout->vertices_per_group),
         intr->src[1].ssa);
      nir_def *record = record_address(
         b, records, index, layout->vertex_stride_B);
      u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
         nir_store_global(
            b, nir_channel(b, intr->src[0].ssa, c),
            nir_iadd_imm(b, record,
                         offset + sizeof(uint32_t) *
                                     (nir_intrinsic_component(intr) + c)),
            .write_mask = 1, .align_mul = 4);
      }
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_store_per_primitive_output: {
      const unsigned slot = nir_intrinsic_io_semantics(intr).location +
                            nir_src_as_uint(intr->src[2]);
      const unsigned offset = slot == VARYING_SLOT_PRIMITIVE_INDICES
         ? layout->index_offset_B : layout->primitive_offset_B[slot];
      nir_def *records = load_param(
         b, params,
         offsetof(struct hk_mesh_emulation_params, primitive_records));
      nir_push_if(
         b, nir_ult_imm(b, intr->src[1].ssa, layout->primitives_per_group));
      nir_def *index = nir_iadd(
         b, nir_imul_imm(b, group, layout->primitives_per_group),
         intr->src[1].ssa);
      nir_def *record = record_address(
         b, records, index, layout->primitive_stride_B);
      u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
         nir_store_global(
            b, nir_channel(b, intr->src[0].ssa, c),
            nir_iadd_imm(b, record,
                         offset + sizeof(uint32_t) *
                                     (nir_intrinsic_component(intr) + c)),
            .write_mask = 1, .align_mul = 4);
      }
      nir_pop_if(b, NULL);
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
   (void)threads;
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_store_shared(b, nir_imm_int(b, 0),
                    nir_imm_int(b, layout->vertex_count),
                    .write_mask = 1, .align_mul = 4);
   nir_store_shared(b, nir_imm_int(b, 0),
                    nir_imm_int(b, layout->primitive_count),
                    .write_mask = 1, .align_mul = 4);
   nir_pop_if(b, NULL);

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
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared | nir_var_mem_global);

   nir_def *params = nir_load_geometry_param_buffer_poly(b);
   nir_def *vertex_records =
      load_param(b, params, offsetof(struct hk_mesh_emulation_params,
                                     vertex_records));
   nir_def *primitive_records =
      load_param(b, params, offsetof(struct hk_mesh_emulation_params,
                                     primitive_records));
   nir_def *expanded_records =
      load_param(b, params, offsetof(struct hk_mesh_emulation_params,
                                     expanded_records));
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
   nir_def *flags =
      load_param_u32(b, params,
                     offsetof(struct hk_mesh_emulation_params, flags), 1);
   nir_def *task_overflow =
      load_param_u32(b, params,
                     offsetof(struct hk_mesh_emulation_params, task_overflow),
                     1);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *group = linear_workgroup_id(b, mesh_grid);
   nir_def *vertex_base =
      nir_imul_imm(b, group, layout->vertices_per_group);
   nir_def *primitive_base =
      nir_imul_imm(b, group, layout->primitives_per_group);
   nir_def *expanded_base = nir_imul_imm(
      b, primitive_base, 3);
   nir_def *primitive_count = nir_load_shared(
      b, 1, 32, nir_imm_int(b, layout->primitive_count), .align_mul = 4);
   nir_def *vertex_count = nir_load_shared(
      b, 1, 32, nir_imm_int(b, layout->vertex_count), .align_mul = 4);
   primitive_count = nir_umin(
      b, primitive_count, nir_imm_int(b, layout->primitives_per_group));
   vertex_count = nir_umin(
      b, vertex_count, nir_imm_int(b, layout->vertices_per_group));
   const uint32_t expected_flags = HK_MESH_EMULATION_FLAG_DEINDEXED |
      (layout->task_payload_size_B ? HK_MESH_EMULATION_FLAG_TASK : 0);
   nir_def *valid = nir_iand(
      b, nir_ieq_imm(b, abi_version, HK_MESH_EMULATION_ABI_VERSION),
      nir_ieq_imm(b, flags, expected_flags));
   if (layout->task_payload_size_B)
      valid = nir_iand(b, valid, nir_ieq_imm(b, task_overflow, 0));
   /* AGX2 ULEA encodes shifts through four. Address 32-byte records as
    * 16-byte records at an even index, keeping this as one grounded ULEA. */
   nir_def *group_record = nir_ulea_agx(
      b, group_records, nir_imul_imm(b, group, 2), nir_imm_int(b, 4));
   /* VkDrawIndexedIndirectCommand is five dwords. Address one GPU-owned draw
    * record per mesh workgroup as (group * 5) << 2. */
   nir_def *draw_record = nir_ulea_agx(
      b, draw_params, nir_imul_imm(b, group, 5), nir_imm_int(b, 2));

   nir_push_if(b, valid);

   for (unsigned base = 0; base < layout->primitives_per_group;
        base += threads) {
      nir_def *primitive = nir_iadd_imm(b, lane, base);
      nir_push_if(b,
                  nir_ult_imm(b, primitive, layout->primitives_per_group));
      nir_def *active = nir_ult(b, primitive, primitive_count);
      nir_def *primitive_index = nir_iadd(b, primitive_base, primitive);
      nir_def *primitive_record = record_address(
         b, primitive_records, primitive_index,
         layout->primitive_stride_B);
      nir_def *indices = nir_load_global(
         b, 3, 32,
         nir_iadd_imm(b, primitive_record, layout->index_offset_B),
         .align_mul = 4);

      for (unsigned c = 0; c < 3; ++c) {
         nir_def *source_vertex = nir_channel(b, indices, c);
         nir_def *source_valid = nir_iand(
            b, active, nir_ult(b, source_vertex, vertex_count));
         nir_def *safe_vertex = nir_umin(
            b, source_vertex,
            nir_imm_int(b, layout->vertices_per_group - 1));
         nir_def *source_index = nir_iadd(b, vertex_base, safe_vertex);
         nir_def *source_record = record_address(
            b, vertex_records, source_index, layout->vertex_stride_B);
         nir_def *expanded = nir_iadd_imm(
            b, nir_imul_imm(b, primitive_index, 3), c);
         nir_def *expanded_record = record_address(
            b, expanded_records, expanded, layout->expanded_stride_B);

         nir_def *position = nir_load_global(
            b, 4, 32,
            nir_iadd_imm(b, source_record, layout->position_offset_B),
            .align_mul = 4);
         position = nir_bcsel(b, source_valid, position,
                              nir_imm_zero(b, 4, 32));
         nir_store_global(
            b, position,
            nir_iadd_imm(b, expanded_record, layout->position_offset_B),
            .write_mask = 0xf, .align_mul = 4);

         for (unsigned slot = VARYING_SLOT_VAR0;
              slot <= VARYING_SLOT_VAR31; ++slot) {
            const unsigned components = MAX2(
               layout->vertex_components[slot],
               layout->primitive_components[slot]);
            if (!components)
               continue;

            const bool per_primitive =
               layout->primitive_components[slot] != 0;
            nir_def *record = per_primitive ? primitive_record
                                            : source_record;
            const unsigned offset = per_primitive
               ? layout->primitive_offset_B[slot]
               : layout->vertex_offset_B[slot];
            nir_def *value = nir_load_global(
               b, components, 32, nir_iadd_imm(b, record, offset),
               .align_mul = 4);
            value = nir_bcsel(b, source_valid, value,
                              nir_imm_zero(b, components, 32));
            nir_store_global(
               b, value,
               nir_iadd_imm(b, expanded_record,
                            layout->expanded_offset_B[slot]),
               .write_mask = BITFIELD_MASK(components), .align_mul = 4);
         }

         nir_def *word = nir_iadd_imm(
            b, nir_imul_imm(b, primitive_index, 3), c);
         nir_def *address = nir_ulea_agx(
            b, index_buffer, word, nir_imm_int(b, 2));
         nir_store_global(b, expanded, address,
                          .write_mask = 1, .align_mul = 4);
      }
      nir_pop_if(b, NULL);
   }

   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_def *expanded_count = nir_imul_imm(b, primitive_count, 3);
   nir_store_global(
      b, nir_vec4(b, expanded_count, primitive_count, expanded_base,
                  primitive_base),
      group_record, .write_mask = 0xf, .align_mul = 16);

   nir_store_global(
      b, nir_vec4(b, expanded_count, nir_imm_int(b, 1), expanded_base,
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

   /* The generic indirect-count predicate copies all five indexed-draw
    * dwords.  The producer above owns the first four; define firstInstance as
    * zero explicitly so draw_params no longer relies on a CPU-side clear. */
   nir_push_if(b, nir_ieq_imm(b, lane, 0));
   nir_store_global(b, nir_imm_int(b, 0),
                    nir_iadd_imm(b, draw_record, 16),
                    .write_mask = 1, .align_mul = 4);
   nir_pop_if(b, NULL);

   nir_barrier(b, .execution_scope = SCOPE_NONE,
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_RELEASE,
               .memory_modes = nir_var_mem_global);
}

static nir_def *
task_record_address(nir_builder *b, nir_def *params, nir_def *task_index,
                    unsigned stride_B)
{
   nir_def *records = load_param(
      b, params, offsetof(struct hk_mesh_emulation_params, task_records));
   return task_record_element_address(b, records, task_index, stride_B);
}

static nir_def *
task_linear_id(nir_builder *b, nir_def *params)
{
   nir_def *grid = load_param_u32(
      b, params, offsetof(struct hk_mesh_emulation_params, task_grid), 3);
   return linear_workgroup_id(b, grid);
}

static bool
lower_task_interface(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct hk_task_emulation_layout *layout = data;
   const bool load = intr->intrinsic == nir_intrinsic_load_task_payload;
   const bool store = intr->intrinsic == nir_intrinsic_store_task_payload;
   const bool launch =
      intr->intrinsic == nir_intrinsic_launch_mesh_workgroups;
   const bool barrier = intr->intrinsic == nir_intrinsic_barrier &&
      (nir_intrinsic_memory_modes(intr) & nir_var_mem_task_payload);
   if (!load && !store && !launch && !barrier)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (barrier) {
      nir_intrinsic_set_memory_modes(
         intr, (nir_intrinsic_memory_modes(intr) &
                ~nir_var_mem_task_payload) |
                  nir_var_mem_global);
      return true;
   }

   nir_def *params = nir_load_geometry_param_buffer_poly(b);
   nir_def *task_index = task_linear_id(b, params);
   nir_def *record = task_record_address(
      b, params, task_index, layout->task_record_stride_B);

   if (load || store) {
      nir_def *records = load_param(
         b, params, offsetof(struct hk_mesh_emulation_params, task_records));
      const unsigned bit_size = store ? intr->src[0].ssa->bit_size
                                      : intr->def.bit_size;
      nir_def *address = task_payload_element_address(
         b, records, task_index, layout->task_record_stride_B,
         intr->src[store ? 1 : 0].ssa,
         HK_MESH_EMULATION_TASK_RECORD_HEADER_SIZE +
            nir_intrinsic_base(intr),
         bit_size);
      if (!address)
         return false;
      if (store) {
         nir_store_global(
            b, intr->src[0].ssa, address,
            .write_mask = nir_intrinsic_write_mask(intr),
            .align_mul = nir_intrinsic_align_mul(intr),
            .align_offset = nir_intrinsic_align_offset(intr),
            .access = nir_intrinsic_access(intr));
         nir_instr_remove(&intr->instr);
      } else {
         nir_def_replace(
            &intr->def,
            nir_load_global(
               b, intr->def.num_components, intr->def.bit_size, address,
               .align_mul = nir_intrinsic_align_mul(intr),
               .align_offset = nir_intrinsic_align_offset(intr),
               .access = nir_intrinsic_access(intr)));
      }
      return true;
   }

   /* Allocate a compact child range.  The experimental cap is never used as
    * partial success: if any task would exceed it, task_overflow is set and
    * every mesh export in this draw writes zero counts.  mesh_grid.x remains
    * bounded so the indirect launch and the O(cap) raster predicate are safe. */
   nir_def *grid = intr->src[0].ssa;
   /* EmitMeshTasksEXT is a workgroup-uniform operation represented in every
    * invocation.  Rendezvous after payload stores, then let lane zero own the
    * single allocation/header/map publication for this task workgroup. */
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
   nir_def *limit = nir_imm_int(
      b, HK_MESH_EMULATION_MAX_EMITTED_GROUPS + 1);
   nir_def *x = nir_umin(b, nir_channel(b, grid, 0), limit);
   nir_def *y = nir_umin(b, nir_channel(b, grid, 1), limit);
   nir_def *z = nir_umin(b, nir_channel(b, grid, 2), limit);
   nir_def *xy = nir_umin(b, nir_imul(b, x, y), limit);
   nir_def *child = nir_umin(b, nir_imul(b, xy, z), limit);
   nir_def *fits_cap = nir_ule_imm(
      b, child, HK_MESH_EMULATION_MAX_EMITTED_GROUPS);
   nir_def *emitted_addr = nir_ulea_agx(
      b, params,
      nir_imm_int(
         b, offsetof(struct hk_mesh_emulation_params,
                     task_emitted_groups) / 4),
      nir_imm_int(b, 2));
   nir_def *first = nir_global_atomic(
      b, 32, emitted_addr, child, .atomic_op = nir_atomic_op_iadd);
   nir_def *end = nir_iadd(b, first, child);
   nir_def *within_cap = nir_iand(
      b, fits_cap,
      nir_ule_imm(b, end, HK_MESH_EMULATION_MAX_EMITTED_GROUPS));
   nir_def *bounded_end = nir_umin(
      b, end, nir_imm_int(b, HK_MESH_EMULATION_MAX_EMITTED_GROUPS));
   nir_def *dispatch_addr = nir_ulea_agx(
      b, params,
      nir_imm_int(b, offsetof(struct hk_mesh_emulation_params, mesh_grid) / 4),
      nir_imm_int(b, 2));
   nir_global_atomic(b, 32, dispatch_addr, bounded_end,
                     .atomic_op = nir_atomic_op_umax);

   nir_push_if(b, nir_inot(b, within_cap));
   nir_def *overflow_addr = nir_ulea_agx(
      b, params,
      nir_imm_int(b,
                  offsetof(struct hk_mesh_emulation_params, task_overflow) / 4),
      nir_imm_int(b, 2));
   nir_global_atomic(b, 32, overflow_addr, nir_imm_int(b, 1),
                     .atomic_op = nir_atomic_op_ior);
   nir_pop_if(b, NULL);

   nir_store_global(b, nir_vec4(b, nir_channel(b, grid, 0),
                                nir_channel(b, grid, 1),
                                nir_channel(b, grid, 2), first),
                    record, .write_mask = 0xf, .align_mul = 16);

   nir_def *map_base = load_param(
      b, params, offsetof(struct hk_mesh_emulation_params, group_to_task));
   nir_variable *cursor = nir_local_variable_create(
      nir_shader_get_entrypoint(b->shader), glsl_uint_type(),
      "task child mapping cursor");
   nir_store_var(b, cursor, nir_imm_int(b, 0), 1);
   nir_def *mapped_children = nir_bcsel(
      b, within_cap, child, nir_imm_int(b, 0));
   nir_push_loop(b);
   nir_def *local = nir_load_var(b, cursor);
   nir_push_if(b, nir_uge(b, local, mapped_children));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_push_if(b, within_cap);
   nir_def *global = nir_iadd(b, first, local);
   nir_def *mapping = record_address(
      b, map_base, global, sizeof(struct hk_mesh_emulation_group_task));
   nir_store_global(b, nir_vec2(b, task_index, local), mapping,
                    .write_mask = 0x3, .align_mul = 8);
   nir_pop_if(b, NULL);
   nir_store_var(b, cursor, nir_iadd_imm(b, local, 1), 1);
   nir_pop_loop(b, NULL);
   nir_pop_if(b, NULL);

   nir_instr_remove(&intr->instr);
   return true;
}

bool
hk_nir_lower_task_emulation(
   nir_shader *shader, struct hk_task_emulation_layout *layout_out)
{
   if (shader->info.stage != MESA_SHADER_TASK ||
       !shader->info.io_lowered || shader->info.workgroup_size_variable ||
       shader->info.task_payload_size >
          HK_MESH_EMULATION_MAX_TASK_PAYLOAD ||
       shader->info.shared_size > 28 * 1024)
      return false;

   struct hk_task_emulation_layout layout = {
      .payload_size_B = shader->info.task_payload_size,
      .task_record_stride_B = ALIGN_POT(
         HK_MESH_EMULATION_TASK_RECORD_HEADER_SIZE +
            shader->info.task_payload_size,
         16),
   };
   uint64_t threads = 1;
   const uint32_t limits[3] = {
      HK_MESH_EMULATION_MAX_WORKGROUP_X,
      HK_MESH_EMULATION_MAX_WORKGROUP_Y,
      HK_MESH_EMULATION_MAX_WORKGROUP_Z,
   };
   for (unsigned i = 0; i < 3; ++i) {
      const uint32_t size = shader->info.workgroup_size[i];
      if (!size || size > limits[i])
         return false;
      layout.local_size[i] = size;
      threads *= size;
   }
   if (threads > HK_MESH_EMULATION_MAX_WORKGROUP_INVOCATIONS)
      return false;

   unsigned launches = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_jump &&
                nir_instr_as_jump(instr)->type == nir_jump_return)
               return false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            launches += intr->intrinsic ==
                        nir_intrinsic_launch_mesh_workgroups;
            if ((intr->intrinsic == nir_intrinsic_load_task_payload ||
                 intr->intrinsic == nir_intrinsic_store_task_payload) &&
                !shader->info.task_payload_size)
               return false;
            if (intr->intrinsic ==
                   nir_intrinsic_launch_mesh_workgroups_with_payload_deref ||
                intr->intrinsic == nir_intrinsic_task_payload_atomic ||
                intr->intrinsic == nir_intrinsic_task_payload_atomic_swap)
               return false;
            if ((intr->intrinsic == nir_intrinsic_load_task_payload &&
                 !task_payload_access_supported(intr, false)) ||
                (intr->intrinsic == nir_intrinsic_store_task_payload &&
                 !task_payload_access_supported(intr, true)))
               return false;
         }
      }
   }
   /* SPIR-V requires EmitMeshTasksEXT in uniform control flow.  The validated
    * Vulkan input therefore supplies that dynamic guarantee; additionally
    * admit exactly one static launch so malformed NIR cannot reserve twice. */
   if (launches != 1)
      return false;

   if (!nir_shader_intrinsics_pass(
          shader, lower_task_interface, nir_metadata_none, &layout))
      return false;

   nir_remove_dead_variables(shader, nir_var_mem_task_payload, NULL);
   shader->info.stage = MESA_SHADER_COMPUTE;
   shader->info.task_payload_size = 0;
   nir_shader_gather_info(shader, nir_shader_get_entrypoint(shader));
   nir_progress(true, nir_shader_get_entrypoint(shader), nir_metadata_none);
   if (layout_out)
      *layout_out = layout;
   return true;
}

bool
hk_nir_lower_mesh_emulation(
   nir_shader *shader, struct hk_mesh_emulation_layout *layout_out)
{
   if (shader->info.stage != MESA_SHADER_MESH ||
       shader->info.shared_size > 32 * 1024)
      return false;

   struct hk_mesh_emulation_layout layout;
   if (!validate_shader(shader, &layout))
      return false;

   const unsigned threads = shader->info.workgroup_size[0] *
                            shader->info.workgroup_size[1] *
                            shader->info.workgroup_size[2];
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   nir_builder prologue = nir_builder_at(nir_before_impl(impl));
   initialize_scratch(&prologue, &layout, threads);

   if (layout.task_payload_size_B) {
      nir_shader_intrinsics_pass(
         shader, lower_mesh_task_derived_ids, nir_metadata_none, NULL);
      nir_shader_intrinsics_pass(
         shader, lower_mesh_task_interface, nir_metadata_none, &layout);
   }

   bool progress = nir_shader_intrinsics_pass(
      shader, lower_output, nir_metadata_none, &layout);
   if (!progress)
      return false;

   nir_builder epilogue = nir_builder_at(nir_after_cf_list(&impl->body));
   export_scratch(&epilogue, &layout, threads);

   /* SPIR-V frontends retain the mesh output variables after nir_lower_io().
    * The emulation pass has replaced every output intrinsic above, so these
    * variables are dead.  Drop them before changing the API stage to compute:
    * nir_shader_gather_info() correctly rejects per-primitive output variables
    * on a compute shader.  Hand-built unit NIR did not carry these variables,
    * which is why the public-pipeline fixture is the regression test here. */
   nir_remove_dead_variables(shader,
                             nir_var_shader_out | nir_var_mem_task_payload,
                             NULL);

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
hk_nir_mesh_emulation_pull_vs(
   const nir_shader_compiler_options *options,
   const struct hk_mesh_emulation_layout *layout)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, options, "G13 software mesh pull VS");
   nir_def *params = nir_load_geometry_param_buffer_poly(&b);
   nir_def *expanded_records =
      load_param(&b, params,
                 offsetof(struct hk_mesh_emulation_params, expanded_records));
   nir_def *vertex = nir_load_vertex_id(&b);
   nir_def *address = record_address(
      &b, expanded_records, vertex, layout->expanded_stride_B);
   nir_def *position = nir_load_global_constant(
      &b, 4, 32, nir_iadd_imm(&b, address, layout->position_offset_B),
      .align_mul = 4,
      .access = ACCESS_CAN_SPECULATE);

   nir_store_output(
      &b, position, nir_imm_int(&b, 0), .write_mask = 0xf,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});

   for (unsigned slot = VARYING_SLOT_VAR0;
        slot <= VARYING_SLOT_VAR31; ++slot) {
      const unsigned components = MAX2(layout->vertex_components[slot],
                                       layout->primitive_components[slot]);
      if (!components)
         continue;

      nir_def *value = nir_load_global_constant(
         &b, components, 32,
         nir_iadd_imm(&b, address, layout->expanded_offset_B[slot]),
         .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
      nir_store_output(
         &b, value, nir_imm_int(&b, 0),
         .write_mask = BITFIELD_MASK(components),
         .src_type = layout->output_type[slot],
         .io_semantics = {.location = slot, .num_slots = 1});
   }

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

   nir_shader *pull_vs = hk_nir_mesh_emulation_pull_vs(options, &layout);
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
