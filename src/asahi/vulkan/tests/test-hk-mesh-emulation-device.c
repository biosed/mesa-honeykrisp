/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Game-free G13 drm-shim fixture for the software-mesh shader pair.  It uses
 * Honeykrisp's real device shader callbacks, code heap and binary cache, but
 * never records or submits a command buffer.
 */

#include "hk_agx3_vdm_decode.h"
#include "hk_device.h"
#include "hk_mesh_emulation_direct.h"
#include "hk_shader.h"
#include "compiler/nir/nir_builder.h"
#include "util/blob.h"
#include "vk_shader.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult result_ = (call);                                               \
      if (result_ != VK_SUCCESS) {                                             \
         fprintf(stderr, "%s: %d\n", #call, result_);                         \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static nir_shader *
triangle_mesh(unsigned payload_size_B)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &options, "G13 device-cache mesh triangle");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   b.shader->info.task_payload_size = payload_size_B;
   b.shader->info.outputs_written =
      VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES |
      VARYING_BIT_VAR(0) | VARYING_BIT_VAR(1);
   b.shader->info.per_primitive_outputs =
      VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_VAR(1);

   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(
      &b, nir_imm_int(&b, 3), nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_store_per_vertex_output(
      &b, nir_vec4(&b, nir_i2f32(&b, lane), nir_imm_float(&b, 0.0f),
                   nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)),
      lane, nir_imm_int(&b, 0), .write_mask = 0xf,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_store_per_vertex_output(
      &b, nir_vec4(&b, nir_i2f32(&b, lane), nir_imm_float(&b, 0.25f),
                   nir_imm_float(&b, 0.75f), nir_imm_float(&b, 1.0f)),
      lane, nir_imm_int(&b, 0), .write_mask = 0xf,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   if (payload_size_B) {
      nir_def *payload = nir_load_task_payload(
         &b, 1, 32, nir_imm_int(&b, 0), .align_mul = 4);
      nir_store_per_vertex_output(
         &b, nir_replicate(&b, payload, 4), lane, nir_imm_int(&b, 0),
         .write_mask = 0xf, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .write_mask = 0x7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   nir_store_per_primitive_output(
      &b, nir_imm_vec4(&b, 1.0f, 0.0f, 0.0f, 1.0f),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 0xf,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
   nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
title_task(void)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &options, "G13 device-cache title task");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.task_payload_size = 256;
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_store_task_payload(&b, lane, nir_imul_imm(&b, lane, 4),
                          .align_mul = 4);
   nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 2, 1, 1));
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static void
check_uploaded_pair(const struct hk_api_shader *obj)
{
   assert(obj->vk.stage == MESA_SHADER_MESH);
   assert(hk_api_num_variants(obj) == HK_MESH_VARIANTS);

   const struct hk_shader *main = &obj->variants[HK_MESH_VARIANT_MAIN];
   const struct hk_shader *rast = &obj->variants[HK_MESH_VARIANT_RAST];
   const struct hk_shader *predicated =
      &obj->variants[HK_MESH_VARIANT_PREDICATED_OBJECT];

   assert(main->compiled && rast->compiled && !predicated->compiled);
   assert(main->info.stage == MESA_SHADER_MESH);
   assert(rast->info.stage == MESA_SHADER_VERTEX);
   assert(main->b.info.stage == MESA_SHADER_COMPUTE);
   assert(rast->b.info.stage == MESA_SHADER_VERTEX);
   assert(main->code_ptr && main->code_size && main->only_linked);
   assert(rast->code_ptr && rast->code_size && !rast->only_linked);
   assert(rast->linked.ht || rast->linked.lazy_ht);
   assert(main->only_linked->b.bo);
   assert(!predicated->code_ptr && !predicated->only_linked);
   assert(main->info.mesh.vertex_stride_B == 32);
   assert(main->info.mesh.primitive_stride_B == 32);
   assert(rast->info.mesh.vertex_stride_B == 48);
   assert(main->info.mesh.vertex_outputs ==
          (VARYING_BIT_POS | VARYING_BIT_VAR(0)));
   assert(main->info.mesh.primitive_outputs ==
          (VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_VAR(1)));
}

static struct hk_api_shader *
deserialize(struct hk_device *dev, const void *data, size_t size)
{
   struct blob_reader reader;
   blob_reader_init(&reader, data, size);
   struct vk_shader *shader = NULL;
   CHECK(dev->vk.shader_ops->deserialize(
      &dev->vk, &reader, hk_shader_binary_version(dev->dev.chip), NULL,
      &shader));
   assert(reader.current == reader.end);
   return container_of(shader, struct hk_api_shader, vk);
}

int
main(void)
{
   setbuf(stdout, NULL);
   assert(getenv("ASAHI_SHIM_CHIP") && getenv("LD_PRELOAD"));

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));

   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   assert(count == 1);

   float priority = 1.0f;
   VkDeviceQueueCreateInfo queue = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue,
   };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   struct hk_device *dev = (void *)vk_device_from_handle(device);
   assert(dev->dev.chip == AGX_CHIP_G13X ||
          dev->dev.chip == AGX_CHIP_G13G);

   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_MESH,
      .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
      .nir = triangle_mesh(0),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *compiled_vk = NULL;
   CHECK(dev->vk.shader_ops->compile(&dev->vk, 1, &info, NULL, NULL, NULL,
                                     &compiled_vk));
   struct hk_api_shader *compiled =
      container_of(compiled_vk, struct hk_api_shader, vk);
   check_uploaded_pair(compiled);

   const uint32_t direct_grid[3] = {2, 1, 1};
   struct hk_mesh_emulation_direct_plan direct;
   assert(hk_mesh_emulation_direct_requirements(
      dev->dev.chip, compiled, direct_grid, &direct));
   assert(direct.draw_count == 2);
   assert(direct.draw_stride_B == sizeof(VkDrawIndexedIndirectCommand));
   assert(direct.draw_params_size_B ==
          2 * sizeof(VkDrawIndexedIndirectCommand));
   assert(hk_mesh_emulation_direct_plan_init(
      dev->dev.chip, compiled, direct_grid, UINT64_C(0x100000), 4096,
      UINT64_C(0x200000), 4096, UINT64_C(0x300000), 4096,
      UINT64_C(0x400000), 4096, UINT64_C(0x500000), 4096,
      UINT64_C(0x600000), 4096, &direct));
   assert(direct.producer == hk_main_mesh_variant(compiled));
   assert(direct.raster == hk_rast_mesh_variant(compiled));
   assert(direct.params.abi_version == HK_MESH_EMULATION_ABI_VERSION);
   assert(direct.params.mesh_grid[0] == 2 &&
          direct.params.mesh_grid[1] == 1 &&
          direct.params.mesh_grid[2] == 1);
   assert(direct.params.primitive_records == UINT64_C(0x200000));
   assert(direct.params.expanded_records == UINT64_C(0x300000));
   assert(direct.params.task_records == 0);
   assert(direct.params.flags == HK_MESH_EMULATION_FLAG_DEINDEXED);
   assert(direct.producer_grid.mode == AGX_CDM_MODE_DIRECT);
   assert(direct.producer_grid.count[0] == 6 &&
          direct.producer_grid.count[1] == 1 &&
          direct.producer_grid.count[2] == 1);
   assert(direct.producer_local_size.x == 3 &&
          direct.producer_local_size.y == 1 &&
          direct.producer_local_size.z == 1);
   assert(direct.barrier == (AGX_BARRIER_ALL | AGX_PREGFX));
   assert(direct.draw.indexed && agx_is_indirect(direct.draw.b));
   assert(direct.draw.index_size == AGX_INDEX_SIZE_U32);
   assert(direct.draw.b.ptr == UINT64_C(0x600000));
   assert(direct.vertex_records_size_B == 192);
   assert(direct.primitive_records_size_B == 64);
   assert(direct.expanded_records_size_B == 288);
   assert(direct.index_buffer_size_B == 24);
   assert(direct.group_records_size_B == 64);
   assert(direct.draw_params_size_B == 40);
   assert(direct.scratch_size_B == 800);
   const uint32_t max_host_grid[3] = {
      HK_G13_MESH_MAX_GROUP_TOTAL, 1, 1,
   };
   assert(hk_mesh_emulation_direct_requirements(
      dev->dev.chip, compiled, max_host_grid, &direct));
   assert(direct.draw_count == HK_G13_MESH_MAX_GROUP_TOTAL);
   const uint32_t over_host_grid[3] = {
      HK_G13_MESH_MAX_GROUP_TOTAL + 1, 1, 1,
   };
   assert(!hk_mesh_emulation_direct_requirements(
      dev->dev.chip, compiled, over_host_grid, &direct));
   const uint32_t product_over_host_grid[3] = {64, 64, 2};
   assert(!hk_mesh_emulation_direct_requirements(
      dev->dev.chip, compiled, product_over_host_grid, &direct));
   const uint32_t thread_overflow_grid[3] = {UINT32_MAX, 1, 1};
   assert(!hk_mesh_emulation_direct_requirements(
      dev->dev.chip, compiled, thread_overflow_grid, &direct));

   struct hk_shader *mutable_main = hk_main_mesh_variant(compiled);
   struct hk_shader *mutable_rast = hk_rast_mesh_variant(compiled);
   const struct hk_mesh_info saved_info = mutable_main->info.mesh;
   const uint32_t saved_rast_stride =
      mutable_rast->info.mesh.vertex_stride_B;
   mutable_main->info.mesh.max_vertices = 256;
   mutable_main->info.mesh.max_primitives = 256;
   mutable_main->info.mesh.vertex_stride_B = 528;
   mutable_main->info.mesh.primitive_stride_B = 528;
   mutable_rast->info.mesh.vertex_stride_B = 528;
   assert(!hk_mesh_emulation_direct_requirements(
      dev->dev.chip, compiled, max_host_grid, &direct));
   mutable_main->info.mesh = saved_info;
   mutable_rast->info.mesh.vertex_stride_B = saved_rast_stride;

   assert(!hk_mesh_emulation_direct_plan_init(
      dev->dev.chip, compiled, direct_grid, UINT64_C(0x100000), 95,
      UINT64_C(0x200000), 4096, UINT64_C(0x300000), 4096,
      UINT64_C(0x400000), 4096, UINT64_C(0x500000), 4096,
      UINT64_C(0x600000), 4096, &direct));
   assert(!hk_mesh_emulation_direct_plan_init(
      AGX_CHIP_G14X, compiled, direct_grid, UINT64_C(0x100000), 4096,
      UINT64_C(0x200000), 4096, UINT64_C(0x300000), 4096,
      UINT64_C(0x400000), 4096, UINT64_C(0x500000), 4096,
      UINT64_C(0x600000), 4096, &direct));

   struct vk_shader_compile_info task_info = {
      .stage = MESA_SHADER_TASK,
      .next_stage_mask = VK_SHADER_STAGE_MESH_BIT_EXT,
      .nir = title_task(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *task_vk = NULL;
   CHECK(dev->vk.shader_ops->compile(&dev->vk, 1, &task_info, NULL, NULL,
                                     NULL, &task_vk));
   struct hk_api_shader *task =
      container_of(task_vk, struct hk_api_shader, vk);
   assert(task->vk.stage == MESA_SHADER_TASK);
   assert(task->variants[0].compiled && task->variants[0].only_linked);
   assert(task->variants[0].info.stage == MESA_SHADER_TASK);
   assert(task->variants[0].b.info.stage == MESA_SHADER_COMPUTE);
   assert(task->variants[0].info.mesh.payload_size_B == 256);
   assert(task->variants[0].info.mesh.local_size[0] == 32);

   struct vk_shader_compile_info payload_mesh_info = {
      .stage = MESA_SHADER_MESH,
      .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
      .nir = triangle_mesh(256),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *payload_mesh_vk = NULL;
   CHECK(dev->vk.shader_ops->compile(&dev->vk, 1, &payload_mesh_info, NULL,
                                     NULL, NULL, &payload_mesh_vk));
   struct hk_api_shader *payload_mesh =
      container_of(payload_mesh_vk, struct hk_api_shader, vk);
   check_uploaded_pair(payload_mesh);
   assert(hk_main_mesh_variant(payload_mesh)->info.mesh.payload_size_B == 256);

   const uint32_t task_grid[3] = {1, 1, 1};
   struct hk_mesh_emulation_direct_plan task_plan;
   assert(hk_mesh_emulation_task_requirements(
      dev->dev.chip, task, payload_mesh, task_grid, &task_plan));
   assert(task_plan.task && task_plan.draw_count == 4096);
   assert(task_plan.task_records_size_B == 272);
   assert(task_plan.group_to_task_size_B == 4096 * 8);
   assert(task_plan.draw_params_size_B ==
          4096 * sizeof(VkDrawIndexedIndirectCommand));
   assert(task_plan.patched_draw_params_size_B ==
          task_plan.draw_params_size_B);
   assert(task_plan.task_grid.mode == AGX_CDM_MODE_DIRECT);
   assert(task_plan.task_grid.count[0] == 32);
   assert(task_plan.task_local_size.x == 32);
   assert(hk_mesh_emulation_task_plan_init(
      dev->dev.chip, task, payload_mesh, task_grid,
      UINT64_C(0x10000000), task_plan.vertex_records_size_B,
      UINT64_C(0x20000000), task_plan.primitive_records_size_B,
      UINT64_C(0x30000000), task_plan.expanded_records_size_B,
      UINT64_C(0x40000000), task_plan.index_buffer_size_B,
      UINT64_C(0x50000000), task_plan.group_records_size_B,
      UINT64_C(0x60000000), task_plan.task_records_size_B,
      UINT64_C(0x70000000), task_plan.group_to_task_size_B,
      UINT64_C(0x80000000), task_plan.draw_params_size_B,
      UINT64_C(0x90000000), task_plan.patched_draw_params_size_B,
      UINT64_C(0xa0000000), &task_plan));
   assert(task_plan.params.mesh_grid[0] == 0 &&
          task_plan.params.mesh_grid[1] == 1 &&
          task_plan.params.mesh_grid[2] == 1);
   assert(task_plan.params.task_grid[0] == 1 &&
          task_plan.params.task_grid[1] == 1 &&
          task_plan.params.task_grid[2] == 1);
   assert(task_plan.params.task_records == UINT64_C(0x60000000));
   assert(task_plan.params.group_to_task == UINT64_C(0x70000000));
   assert(task_plan.params.task_record_stride_B == 272);
   assert(task_plan.params.task_payload_size_B == 256);
   assert(task_plan.params.flags ==
          (HK_MESH_EMULATION_FLAG_DEINDEXED |
           HK_MESH_EMULATION_FLAG_TASK));
   assert(task_plan.draw.b.ptr == UINT64_C(0x90000000));
   assert(task_plan.producer_grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL);
   assert(task_plan.producer_grid.ptr ==
          UINT64_C(0xa0000000) +
             offsetof(struct hk_mesh_emulation_params, mesh_grid));
   assert(!hk_mesh_emulation_task_requirements(
      AGX_CHIP_G14X, task, payload_mesh, task_grid, &task_plan));

   struct blob task_blob;
   blob_init(&task_blob);
   assert(task->vk.ops->serialize(&dev->vk, &task->vk, &task_blob));
   assert(!task_blob.out_of_memory && task_blob.size > 0);
   struct hk_api_shader *task_clone =
      deserialize(dev, task_blob.data, task_blob.size);
   assert(task_clone->vk.stage == MESA_SHADER_TASK);
   assert(task_clone->variants[0].compiled &&
          task_clone->variants[0].only_linked);
   assert(task_clone->variants[0].info.mesh.payload_size_B == 256);
   assert(task_clone->variants[0].code_size == task->variants[0].code_size);
   assert(memcmp(task_clone->variants[0].code_ptr,
                 task->variants[0].code_ptr,
                 task->variants[0].code_size) == 0);

   struct blob blob;
   blob_init(&blob);
   assert(compiled->vk.ops->serialize(&dev->vk, &compiled->vk, &blob));
   assert(!blob.out_of_memory && blob.size > 0);

   unsigned cache_entries = 0;
   for (unsigned i = 0; i < HK_SHADER_BINARY_CACHE_MAX; ++i)
      cache_entries += dev->shader_binary_cache.entries[i] != NULL;
   assert(cache_entries < HK_SHADER_BINARY_CACHE_MAX);

   struct hk_api_shader *first = deserialize(dev, blob.data, blob.size);
   struct hk_api_shader *second = deserialize(dev, blob.data, blob.size);
   check_uploaded_pair(first);
   check_uploaded_pair(second);

   /* Both wrappers borrow one immutable CPU payload from the device cache,
    * but own separate mutable linked uploads and remain independently valid. */
   assert(first != second);
   assert(first->binary_cache_entry);
   assert(first->binary_cache_entry == second->binary_cache_entry);
   assert(hk_main_mesh_variant(first)->code_ptr ==
          hk_main_mesh_variant(second)->code_ptr);
   assert(hk_rast_mesh_variant(first)->code_ptr ==
          hk_rast_mesh_variant(second)->code_ptr);
   assert(hk_main_mesh_variant(first)->code_size ==
          hk_main_mesh_variant(compiled)->code_size);
   assert(hk_rast_mesh_variant(first)->code_size ==
          hk_rast_mesh_variant(compiled)->code_size);
   assert(memcmp(hk_main_mesh_variant(first)->code_ptr,
                 hk_main_mesh_variant(compiled)->code_ptr,
                 hk_main_mesh_variant(first)->code_size) == 0);
   assert(memcmp(hk_rast_mesh_variant(first)->code_ptr,
                 hk_rast_mesh_variant(compiled)->code_ptr,
                 hk_rast_mesh_variant(first)->code_size) == 0);
   assert(hk_main_mesh_variant(first)->only_linked !=
          hk_main_mesh_variant(second)->only_linked);
   assert(!hk_rast_mesh_variant(first)->only_linked);
   assert(!hk_rast_mesh_variant(second)->only_linked);
   assert(hk_rast_mesh_variant(first)->linked.lazy_ht);
   assert(hk_rast_mesh_variant(second)->linked.lazy_ht);
   assert(&hk_rast_mesh_variant(first)->linked !=
          &hk_rast_mesh_variant(second)->linked);

   const uint32_t producer_size = hk_main_mesh_variant(compiled)->code_size;
   const uint32_t pull_vs_size = hk_rast_mesh_variant(compiled)->code_size;
   compiled->vk.ops->destroy(&dev->vk, &compiled->vk, NULL);
   first->vk.ops->destroy(&dev->vk, &first->vk, NULL);
   /* The cache and second wrapper still own the shared bytes. */
   assert(hk_main_mesh_variant(second)->code_ptr);
   assert(hk_rast_mesh_variant(second)->code_ptr);
   second->vk.ops->destroy(&dev->vk, &second->vk, NULL);
   task->vk.ops->destroy(&dev->vk, &task->vk, NULL);
   task_clone->vk.ops->destroy(&dev->vk, &task_clone->vk, NULL);
   payload_mesh->vk.ops->destroy(&dev->vk, &payload_mesh->vk, NULL);

   blob_finish(&task_blob);
   blob_finish(&blob);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);

   printf("G13 mesh device/cache: uploaded MAIN %u B, RAST %u B, "
          "binary cache hit PASS\n",
          producer_size, pull_vs_size);
   return 0;
}
