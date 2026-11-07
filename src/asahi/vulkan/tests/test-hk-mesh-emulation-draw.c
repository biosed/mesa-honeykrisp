/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Physical, game-free G13 software-mesh triangle. The public mesh extension
 * remains disabled: this fixture compiles/binds the admitted pair through
 * Honeykrisp's internal API, submits one direct mesh workgroup, and verifies
 * the resulting RGBA8 image through an ordinary transfer readback.
 */

#undef NDEBUG
#include <assert.h>

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_nir_lower_mesh_emulation.h"
#include "hk_shader.h"
#include "compiler/nir/nir_builder.h"
#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"

#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define WIDTH  64
#define HEIGHT 64

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult result_ = (call);                                               \
      if (result_ != VK_SUCCESS) {                                             \
         fprintf(stderr, "%s: %d\n", #call, result_);                        \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory_properties;

typedef bool (*hk_test_mesh_emulation_draw_fn)(
   VkCommandBuffer command, struct hk_api_shader *mesh,
   const uint32_t group_count[3], uint16_t draw_id,
   struct hk_mesh_emulation_record_trace *trace);

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((bits & BITFIELD_BIT(i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }

   fprintf(stderr, "no memory type for flags 0x%x\n", flags);
   exit(2);
}

static VkDeviceMemory
allocate_memory(VkMemoryRequirements req, VkMemoryPropertyFlags flags)
{
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, flags),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(device, &info, NULL, &memory));
   return memory;
}

static nir_shader *
triangle_mesh(void)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &options, "G13 submitted mesh triangle");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   b.shader->info.outputs_written =
      VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES;
   b.shader->info.per_primitive_outputs = VARYING_BIT_PRIMITIVE_INDICES;

   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *lane0 = nir_ieq_imm(&b, lane, 0);
   nir_def *lane1 = nir_ieq_imm(&b, lane, 1);
   nir_def *lane2 = nir_ieq_imm(&b, lane, 2);
   nir_def *x = nir_bcsel(
      &b, lane0, nir_imm_float(&b, -0.75f),
      nir_bcsel(&b, lane1, nir_imm_float(&b, 0.75f),
                nir_imm_float(&b, 0.0f)));
   nir_def *y = nir_bcsel(&b, lane2, nir_imm_float(&b, 0.75f),
                          nir_imm_float(&b, -0.75f));

   nir_set_vertex_and_primitive_count(
      &b, nir_imm_int(&b, 3), nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_store_per_vertex_output(
      &b, nir_vec4(&b, x, y, nir_imm_float(&b, 0.0f),
                   nir_imm_float(&b, 1.0f)),
      lane, nir_imm_int(&b, 0), .write_mask = 0xf,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_push_if(&b, lane0);
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .write_mask = 0x7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static void
image_barrier(VkCommandBuffer command, VkImage image,
              VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
              VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
              VkImageLayout old_layout, VkImageLayout new_layout)
{
   const VkImageMemoryBarrier2 image_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &image_barrier,
   };
   vkCmdPipelineBarrier2(command, &dependency);
}

int
main(void)
{
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));

   uint32_t physical_count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &physical_count, &physical));
   assert(physical_count == 1);
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = true,
      .synchronization2 = true,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
   struct hk_device *dev = (void *)vk_device_from_handle(device);
   assert(dev->dev.chip == AGX_CHIP_G13G ||
          dev->dev.chip == AGX_CHIP_G13X);

   const char *icd_path = getenv("HK_TEST_ICD");
   assert(icd_path != NULL);
   void *icd = dlopen(icd_path, RTLD_NOW | RTLD_LOCAL);
   if (!icd) {
      fprintf(stderr, "dlopen(%s): %s\n", icd_path, dlerror());
      return 2;
   }
   hk_test_mesh_emulation_draw_fn record_mesh =
      (hk_test_mesh_emulation_draw_fn)dlsym(
         icd, "hk_test_mesh_emulation_draw");
   if (!record_mesh) {
      fprintf(stderr, "mesh test hook: %s\n", dlerror());
      return 2;
   }

   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
   };
   VkImage image;
   CHECK(vkCreateImage(device, &image_info, NULL, &image));
   VkMemoryRequirements image_req;
   vkGetImageMemoryRequirements(device, image, &image_req);
   VkDeviceMemory image_memory =
      allocate_memory(image_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   CHECK(vkBindImageMemory(device, image, image_memory, 0));

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView view;
   CHECK(vkCreateImageView(device, &view_info, NULL, &view));

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = WIDTH * HEIGHT * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer readback;
   CHECK(vkCreateBuffer(device, &buffer_info, NULL, &readback));
   VkMemoryRequirements buffer_req;
   vkGetBufferMemoryRequirements(device, readback, &buffer_req);
   VkDeviceMemory buffer_memory = allocate_memory(
      buffer_req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   CHECK(vkBindBufferMemory(device, readback, buffer_memory, 0));
   uint8_t *pixels;
   CHECK(vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0,
                     (void **)&pixels));
   memset(pixels, 0xcd, WIDTH * HEIGHT * 4);

   const VkShaderModuleCreateInfo vertex_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_vert_spv),
      .pCode = g17p_wsi_triangle_vert_spv,
   };
   const VkShaderModuleCreateInfo fragment_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_frag_spv),
      .pCode = g17p_wsi_triangle_frag_spv,
   };
   VkShaderModule vertex_module, fragment_module;
   CHECK(vkCreateShaderModule(device, &vertex_module_info, NULL,
                              &vertex_module));
   CHECK(vkCreateShaderModule(device, &fragment_module_info, NULL,
                              &fragment_module));

   VkPipelineLayout layout;
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   CHECK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));

   const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkViewport viewport = {0, 0, WIDTH, HEIGHT, 0, 1};
   const VkRect2D scissor = {{0, 0}, {WIDTH, HEIGHT}};
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &(VkPipelineRenderingCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
         .colorAttachmentCount = 1,
         .pColorAttachmentFormats = &color_format,
      },
      .stageCount = 2,
      .pStages = (VkPipelineShaderStageCreateInfo[]){
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vertex_module,
          .pName = "main"},
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fragment_module,
          .pName = "main"},
      },
      .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      },
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      },
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
         .viewportCount = 1,
         .pViewports = &viewport,
         .scissorCount = 1,
         .pScissors = &scissor,
      },
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL,
         .cullMode = VK_CULL_MODE_NONE,
         .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
         .lineWidth = 1.0f,
      },
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      },
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
         .attachmentCount = 1,
         .pAttachments = &(VkPipelineColorBlendAttachmentState){
            .colorWriteMask = 0xf,
         },
      },
      .layout = layout,
   };
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline));

   struct vk_shader_compile_info mesh_info = {
      .stage = MESA_SHADER_MESH,
      .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
      .nir = triangle_mesh(),
      .robustness = &vk_robustness_disabled,
   };
   struct vk_shader *mesh_vk = NULL;
   CHECK(dev->vk.shader_ops->compile(&dev->vk, 1, &mesh_info, NULL, NULL,
                                     NULL, &mesh_vk));
   struct hk_api_shader *mesh =
      container_of(mesh_vk, struct hk_api_shader, vk);

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
   const VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command;
   CHECK(vkAllocateCommandBuffers(device, &command_info, &command));
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(command, &begin_info));
   image_barrier(command, image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

   const VkRenderingAttachmentInfo attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}},
   };
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {WIDTH, HEIGHT}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachment,
   };
   vkCmdBeginRendering(command, &rendering);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(command);
   if (cmd->vk.record_result != VK_SUCCESS) {
      fprintf(stderr, "begin rendering failed: VkResult=%d\n",
              cmd->vk.record_result);
      return 2;
   }
   struct hk_mesh_emulation_record_trace trace = {
      .producer_root = UINT64_MAX,
      .geometry_params = UINT64_MAX,
      .mesh_grid_addr = UINT64_MAX,
      .linked_raster = UINTPTR_MAX,
   };
   const uint32_t zero_grid[3] = {1, 0, 1};
   assert(record_mesh(command, mesh, zero_grid, 0, &trace));
   assert(trace.producer_root == 0 && trace.geometry_params == 0 &&
          trace.mesh_grid_addr == 0 && trace.linked_raster == 0);
   assert(!vk_command_buffer_has_error(&cmd->vk));

   const uint32_t grid[3] = {1, 1, 1};
   if (!record_mesh(command, mesh, grid, 0, &trace)) {
      fprintf(stderr, "G13 mesh recording failed: VkResult=%d root=%#" PRIx64
                      " params=%#" PRIx64 "\n",
              cmd->vk.record_result, cmd->state.gfx.root,
              cmd->state.gfx.descriptors.root.draw.geometry_params);
      return 2;
   }
   assert(trace.producer_root != 0 && trace.geometry_params != 0);
   assert(trace.linked_raster != 0);
   assert(trace.linked_raster ==
          (uintptr_t)cmd->state.gfx.linked[MESA_SHADER_VERTEX]);
   assert(trace.producer_root == cmd->state.gfx.root);
   assert(cmd->state.gfx.root_cpu != NULL);
   assert(trace.geometry_params ==
          cmd->state.gfx.root_cpu->draw.geometry_params);
   assert(trace.mesh_grid_addr ==
          trace.geometry_params +
             offsetof(struct hk_mesh_emulation_params, mesh_grid));
   assert(trace.mesh_grid_addr == cmd->state.gfx.root_cpu->mesh_grid_addr);

   vkCmdEndRendering(command);
   image_barrier(command, image,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   const VkBufferImageCopy copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command, image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback, 1, &copy);
   CHECK(vkEndCommandBuffer(command));

   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned orange = 0, black = 0, other = 0;
   for (unsigned i = 0; i < WIDTH * HEIGHT; ++i) {
      const uint8_t *p = pixels + i * 4;
      if (p[0] > 240 && p[1] > 55 && p[1] < 90 && p[2] < 30 && p[3] > 240)
         orange++;
      else if (p[0] < 8 && p[1] < 8 && p[2] < 8 && p[3] > 240)
         black++;
      else
         other++;
   }
   const uint8_t *center = pixels + ((HEIGHT / 2) * WIDTH + WIDTH / 2) * 4;
   const bool pass = orange > 500 && black > 500 && other == 0 &&
                     center[0] > 240 && center[1] > 55 && center[1] < 90;
   printf("G13 mesh submitted triangle: orange=%u black=%u other=%u "
          "center=%u,%u,%u,%u %s\n",
          orange, black, other, center[0], center[1], center[2], center[3],
          pass ? "PASS" : "FAIL");

   mesh->vk.ops->destroy(&dev->vk, &mesh->vk, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, vertex_module, NULL);
   vkDestroyShaderModule(device, fragment_module, NULL);
   vkDestroyBuffer(device, readback, NULL);
   vkFreeMemory(device, buffer_memory, NULL);
   vkDestroyImageView(device, view, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, image_memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(icd);
   return pass ? 0 : 1;
}
