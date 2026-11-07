/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_mesh_emulation_direct.h"

#include <limits.h>
#include <string.h>

#include "hk_shader.h"

bool
hk_mesh_emulation_direct_plan_init(
   enum agx_chip chip, const struct hk_api_shader *api,
   const uint32_t group_count[3], uint64_t vertex_records,
   uint64_t vertex_records_size_B, uint64_t index_buffer,
   uint64_t index_buffer_size_B, uint64_t group_records,
   uint64_t group_records_size_B, uint64_t draw_params,
   struct hk_mesh_emulation_direct_plan *plan)
{
   if (!api || !group_count || !plan ||
       (chip != AGX_CHIP_G13G && chip != AGX_CHIP_G13X) ||
       api->vk.stage != MESA_SHADER_MESH)
      return false;

   const struct hk_shader *producer =
      &api->variants[HK_MESH_VARIANT_MAIN];
   const struct hk_shader *raster =
      &api->variants[HK_MESH_VARIANT_RAST];
   const struct hk_shader *predicated =
      &api->variants[HK_MESH_VARIANT_PREDICATED_OBJECT];
   if (!producer->compiled || !producer->only_linked ||
       producer->b.info.stage != MESA_SHADER_COMPUTE ||
       !raster->compiled || !raster->only_linked ||
       raster->b.info.stage != MESA_SHADER_VERTEX || predicated->compiled ||
       !producer->info.mesh.max_vertices ||
       !producer->info.mesh.max_primitives ||
       producer->info.mesh.prim != MESA_PRIM_TRIANGLES)
      return false;

   uint64_t groups = 1;
   uint32_t threads[3];
   for (unsigned i = 0; i < 3; ++i) {
      const uint32_t local = producer->info.mesh.local_size[i];
      if (!group_count[i] || !local ||
          group_count[i] > UINT32_MAX / local ||
          groups > UINT64_MAX / group_count[i])
         return false;

      threads[i] = group_count[i] * local;
      groups *= group_count[i];
   }

   const uint64_t vertices =
      groups * producer->info.mesh.max_vertices;
   const uint64_t primitives =
      groups * producer->info.mesh.max_primitives;
   if (groups &&
       (vertices / groups != producer->info.mesh.max_vertices ||
        primitives / groups != producer->info.mesh.max_primitives))
      return false;

   if (vertices > UINT64_MAX / 16 || primitives > UINT32_MAX / 3 ||
       groups > UINT64_MAX / sizeof(struct hk_mesh_emulation_group_record))
      return false;

   const uint64_t vertex_need = vertices * 16;
   const uint64_t index_count = primitives * 3;
   const uint64_t index_need = index_count * sizeof(uint32_t);
   const uint64_t group_need =
      groups * sizeof(struct hk_mesh_emulation_group_record);
   if (!vertex_records || (vertex_records & 15) || !index_buffer ||
       (index_buffer & 3) || !group_records || (group_records & 31) ||
       !draw_params || (draw_params & 3) ||
       vertex_need > vertex_records_size_B ||
       index_need > index_buffer_size_B || index_buffer_size_B > UINT32_MAX ||
       group_need > group_records_size_B)
      return false;

   memset(plan, 0, sizeof(*plan));
   plan->producer = producer;
   plan->raster = raster;
   plan->params = (struct hk_mesh_emulation_params){
      .vertex_records = vertex_records,
      .index_buffer = index_buffer,
      .group_records = group_records,
      .draw_params = draw_params,
      .mesh_grid = {group_count[0], group_count[1], group_count[2]},
      .abi_version = HK_MESH_EMULATION_ABI_VERSION,
   };
   plan->producer_grid = agx_3d(threads[0], threads[1], threads[2]);
   plan->producer_local_size = agx_workgroup(
      producer->info.mesh.local_size[0], producer->info.mesh.local_size[1],
      producer->info.mesh.local_size[2]);
   plan->barrier = AGX_BARRIER_ALL | AGX_PREGFX;
   plan->draw = agx_draw_indexed_indirect(
      draw_params, index_buffer, (uint32_t)index_buffer_size_B,
      AGX_INDEX_SIZE_U32, false);
   plan->vertex_records_size_B = vertex_need;
   plan->index_buffer_size_B = index_need;
   plan->group_records_size_B = group_need;
   return true;
}
