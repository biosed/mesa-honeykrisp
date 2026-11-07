/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public-API G13 task -> payload -> mesh physical readback fixture.  The
 * --record-only mode is safe for drm-shim and never submits a command buffer.
 */

#undef NDEBUG
#include <assert.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static const uint32_t task_spv[] = {
#include "g13_mesh_task_readback.task.spv.h"
};

static const uint32_t mesh_spv[] = {
#include "g13_mesh_task_readback.mesh.spv.h"
};

static const uint32_t fragment_spv[] = {
#include "g13_mesh_task_readback.frag.spv.h"
};

#define WIDTH 64u
#define HEIGHT 64u
#define CASE_COUNT 3u
#define IMAGE_SIZE (WIDTH * HEIGHT * 4u)

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult result_ = (call);                                               \
      if (result_ != VK_SUCCESS) {                                             \
         fprintf(stderr, "%s: %d\n", #call, result_);                        \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

struct target {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view;
};

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
};

static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory_properties;

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }

   fprintf(stderr, "no memory type for flags %#x\n", flags);
   exit(2);
}

static VkDeviceMemory
allocate_memory(VkMemoryRequirements requirements,
                VkMemoryPropertyFlags flags)
{
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(requirements.memoryTypeBits, flags),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(device, &info, NULL, &memory));
   return memory;
}

static struct buffer
create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
              VkMemoryPropertyFlags flags, bool map)
{
   struct buffer buffer = {0};
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   CHECK(vkCreateBuffer(device, &info, NULL, &buffer.buffer));
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
   buffer.memory = allocate_memory(requirements, flags);
   CHECK(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0));
   if (map)
      CHECK(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0,
                        &buffer.map));
   return buffer;
}

static struct target
create_target(void)
{
   struct target target = {0};
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
   CHECK(vkCreateImage(device, &image_info, NULL, &target.image));
   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, target.image, &requirements);
   target.memory =
      allocate_memory(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   CHECK(vkBindImageMemory(device, target.image, target.memory, 0));

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = target.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   CHECK(vkCreateImageView(device, &view_info, NULL, &target.view));
   return target;
}

static void
image_barrier(VkCommandBuffer command, VkImage image,
              VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
              VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
              VkImageLayout old_layout, VkImageLayout new_layout)
{
   const VkImageMemoryBarrier2 barrier = {
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
      .pImageMemoryBarriers = &barrier,
   };
   vkCmdPipelineBarrier2(command, &dependency);
}

static void
buffer_barrier(VkCommandBuffer command, VkBuffer buffer,
               VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
               VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
               VkDeviceSize size)
{
   const VkBufferMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer,
      .offset = 0,
      .size = size,
   };
   const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
   };
   vkCmdPipelineBarrier2(command, &dependency);
}

static VkShaderModule
shader_module(const uint32_t *code, size_t size)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkShaderModule module;
   CHECK(vkCreateShaderModule(device, &info, NULL, &module));
   return module;
}

static void
begin_target(VkCommandBuffer command, const struct target *target,
             VkPipeline pipeline)
{
   image_barrier(command, target->image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
   const VkRenderingAttachmentInfo attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = target->view,
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
}

static void
end_target(VkCommandBuffer command, const struct target *target,
           VkBuffer readback, VkDeviceSize offset)
{
   vkCmdEndRendering(command);
   image_barrier(command, target->image,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   const VkBufferImageCopy copy = {
      .bufferOffset = offset,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command, target->image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback, 1, &copy);
}

static unsigned
quarter_unorm(unsigned quarter)
{
   assert(quarter >= 1 && quarter <= 3);
   static const uint8_t values[] = {0, 64, 128, 191};
   return values[quarter];
}

static bool
near_byte(uint8_t actual, unsigned expected)
{
   const int delta = (int)actual - (int)expected;
   return delta >= -1 && delta <= 1;
}

static bool
check_grid(const uint8_t *pixels, unsigned task_count, const char *name)
{
   const unsigned columns = task_count * 2;
   unsigned wrong = 0;
   for (unsigned y = 0; y < HEIGHT; ++y) {
      const unsigned child_y = ((2 * y + 1) * 2) / (2 * HEIGHT);
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned column = ((2 * x + 1) * columns) / (2 * WIDTH);
         const unsigned task = column / 2;
         const unsigned child_x = column & 1;
         const uint8_t *pixel = pixels + (y * WIDTH + x) * 4;
         wrong += !near_byte(pixel[0], quarter_unorm(task + 1)) ||
                  !near_byte(pixel[1], quarter_unorm(child_x ? 3 : 1)) ||
                  !near_byte(pixel[2], quarter_unorm(child_y ? 3 : 1)) ||
                  pixel[3] != 255;
      }
   }

   printf("%s: task=%u child={2,2,1} exact_regions=%u wrong=%u %s\n",
          name, task_count, WIDTH * HEIGHT - wrong, wrong,
          wrong == 0 ? "PASS" : "FAIL");
   return wrong == 0;
}

static bool
check_black(const uint8_t *pixels)
{
   unsigned wrong = 0;
   for (unsigned i = 0; i < WIDTH * HEIGHT; ++i) {
      const uint8_t *pixel = pixels + i * 4;
      wrong += pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0 ||
               pixel[3] != 255;
   }

   printf("aggregate overflow {513,1,1} x child {2,2,1}: "
          "black=%u wrong=%u %s\n",
          WIDTH * HEIGHT - wrong, wrong, wrong == 0 ? "PASS" : "FAIL");
   return wrong == 0;
}

int
main(int argc, char **argv)
{
   const bool record_only = argc == 2 &&
                            strcmp(argv[1], "--record-only") == 0;
   const bool submit = argc == 2 && strcmp(argv[1], "--submit") == 0;
   if (!record_only && !submit) {
      fprintf(stderr, "usage: %s --record-only|--submit\n", argv[0]);
      return 2;
   }

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
   assert(physical_count == 1 && physical != VK_NULL_HANDLE);
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   VkPhysicalDeviceMeshShaderFeaturesEXT available_mesh = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_mesh,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available);
   assert(available_mesh.meshShader && available_mesh.taskShader);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = true,
      .synchronization2 = true,
   };
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .pNext = &features13,
      .taskShader = true,
      .meshShader = true,
   };
   const char *extensions[] = {VK_EXT_MESH_SHADER_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &mesh_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   CHECK(vkCreateDevice(physical, &device_info, NULL, &device));

   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);
   PFN_vkCmdDrawMeshTasksEXT draw_mesh =
      (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(
         device, "vkCmdDrawMeshTasksEXT");
   PFN_vkCmdDrawMeshTasksIndirectEXT draw_mesh_indirect =
      (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(
         device, "vkCmdDrawMeshTasksIndirectEXT");
   assert(draw_mesh != NULL && draw_mesh_indirect != NULL);

   VkShaderModule task_module = shader_module(task_spv, sizeof(task_spv));
   VkShaderModule mesh_module = shader_module(mesh_spv, sizeof(mesh_spv));
   VkShaderModule fragment_module =
      shader_module(fragment_spv, sizeof(fragment_spv));
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));

   const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineRenderingCreateInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &color_format,
   };
   const VkViewport viewport = {0, 0, WIDTH, HEIGHT, 0, 1};
   const VkRect2D scissor = {{0, 0}, {WIDTH, HEIGHT}};
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering_info,
      .stageCount = 3,
      .pStages = (VkPipelineShaderStageCreateInfo[]){
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
          .module = task_module,
          .pName = "main"},
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
          .module = mesh_module,
          .pName = "main"},
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fragment_module,
          .pName = "main"},
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

   struct target targets[CASE_COUNT];
   for (unsigned i = 0; i < CASE_COUNT; ++i)
      targets[i] = create_target();
   struct buffer readback = create_buffer(
      CASE_COUNT * IMAGE_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      true);
   memset(readback.map, 0xcd, CASE_COUNT * IMAGE_SIZE);
   struct buffer indirect = create_buffer(
      2 * sizeof(VkDrawMeshTasksIndirectCommandEXT),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);

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

   const VkDrawMeshTasksIndirectCommandEXT indirect_record = {3, 1, 1};
   /* Each task's four-child EmitMeshTasksEXT is below the advertised
    * per-workgroup total.  513 task groups produce 2052 aggregate children,
    * exercising the experimental whole-draw 2048-cap sanitizer with an
    * otherwise API-valid draw. */
   const VkDrawMeshTasksIndirectCommandEXT overflow_record = {513, 1, 1};
   vkCmdUpdateBuffer(command, indirect.buffer, 0, sizeof(indirect_record),
                     &indirect_record);
   vkCmdUpdateBuffer(command, indirect.buffer, sizeof(indirect_record),
                     sizeof(overflow_record), &overflow_record);
   buffer_barrier(command, indirect.buffer,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                  VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                  2 * sizeof(VkDrawMeshTasksIndirectCommandEXT));

   begin_target(command, &targets[0], pipeline);
   draw_mesh(command, 2, 1, 1);
   end_target(command, &targets[0], readback.buffer, 0);

   begin_target(command, &targets[1], pipeline);
   draw_mesh_indirect(command, indirect.buffer, 0, 1,
                      sizeof(VkDrawMeshTasksIndirectCommandEXT));
   end_target(command, &targets[1], readback.buffer, IMAGE_SIZE);

   begin_target(command, &targets[2], pipeline);
   draw_mesh_indirect(command, indirect.buffer, sizeof(indirect_record), 1,
                      sizeof(VkDrawMeshTasksIndirectCommandEXT));
   end_target(command, &targets[2], readback.buffer, 2 * IMAGE_SIZE);
   buffer_barrier(command, readback.buffer,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_HOST_BIT,
                  VK_ACCESS_2_HOST_READ_BIT,
                  CASE_COUNT * IMAGE_SIZE);
   CHECK(vkEndCommandBuffer(command));

   bool pass = true;
   if (record_only) {
      puts("G13 task mesh public readback: record-only PASS (no submit)");
   } else {
      const VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &command,
      };
      CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));
      const uint8_t *pixels = readback.map;
      pass &= check_grid(pixels, 2, "direct {2,1,1}");
      pass &= check_grid(pixels + IMAGE_SIZE, 3,
                         "GPU-written indirect {3,1,1}");
      pass &= check_black(pixels + 2 * IMAGE_SIZE);
   }

   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyBuffer(device, indirect.buffer, NULL);
   vkFreeMemory(device, indirect.memory, NULL);
   vkUnmapMemory(device, readback.memory);
   vkDestroyBuffer(device, readback.buffer, NULL);
   vkFreeMemory(device, readback.memory, NULL);
   for (unsigned i = 0; i < CASE_COUNT; ++i) {
      vkDestroyImageView(device, targets[i].view, NULL);
      vkDestroyImage(device, targets[i].image, NULL);
      vkFreeMemory(device, targets[i].memory, NULL);
   }
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, fragment_module, NULL);
   vkDestroyShaderModule(device, mesh_module, NULL);
   vkDestroyShaderModule(device, task_module, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return pass ? 0 : 1;
}
