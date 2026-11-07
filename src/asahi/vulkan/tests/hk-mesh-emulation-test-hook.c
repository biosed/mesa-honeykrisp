/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Build-test-only bridge into the real ICD command buffer. This source is
 * omitted from non-test builds, and is not a Vulkan entrypoint.
 */

#include "hk_cmd_buffer.h"
#include "hk_nir_lower_mesh_emulation.h"
#include "hk_shader.h"

__attribute__((visibility("default"))) bool hk_test_mesh_root_snapshot(
   VkCommandBuffer command, bool task, uint64_t *geometry_params,
   uint64_t *mesh_grid_addr);

__attribute__((visibility("default"))) bool
hk_test_mesh_root_snapshot(VkCommandBuffer command, bool task,
                           uint64_t *geometry_params,
                           uint64_t *mesh_grid_addr)
{
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(command);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (!gfx->root_cpu)
      return false;

   const uint64_t params = gfx->root_cpu->draw.geometry_params;
   const uint64_t grid = params +
      (task ? offsetof(struct hk_mesh_emulation_params, task_grid)
            : offsetof(struct hk_mesh_emulation_params, mesh_grid));
   if (geometry_params)
      *geometry_params = params;
   if (mesh_grid_addr)
      *mesh_grid_addr = gfx->root_cpu->mesh_grid_addr;

   return params != 0 && gfx->descriptors.root.draw.geometry_params == params &&
          gfx->descriptors.root.mesh_grid_addr == grid &&
          gfx->root_cpu->mesh_grid_addr == grid;
}

__attribute__((visibility("default"))) bool hk_test_mesh_emulation_draw(
   VkCommandBuffer command, struct hk_api_shader *mesh,
   const uint32_t group_count[3], uint16_t draw_id,
   struct hk_mesh_emulation_record_trace *trace);

__attribute__((visibility("default"))) bool
hk_test_mesh_emulation_draw(
   VkCommandBuffer command, struct hk_api_shader *mesh,
   const uint32_t group_count[3], uint16_t draw_id,
   struct hk_mesh_emulation_record_trace *trace)
{
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(command);
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_VERTEX, NULL);
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_MESH, mesh);
   return hk_cmd_draw_mesh_g13_direct(cmd, group_count, draw_id, trace);
}
