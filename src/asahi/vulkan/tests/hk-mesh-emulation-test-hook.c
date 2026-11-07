/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Build-test-only bridge into the real ICD command buffer. This source is
 * omitted from non-test builds, and is not a Vulkan entrypoint.
 */

#include "hk_cmd_buffer.h"
#include "hk_shader.h"

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
