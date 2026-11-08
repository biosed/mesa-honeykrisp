/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public-API, no-submit G13 software-mesh fixture.  The G13X drm-shim path
 * creates a real mesh+fragment pipeline and records vkCmdDrawMeshTasksEXT.
 * The G14G path proves that the extension/features/properties stay absent.
 */

#undef NDEBUG
#include <assert.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static const uint32_t g13_mesh_public_triangle_spv[] = {
#include "g13_mesh_public_triangle.spv.h"
};

static const uint32_t g13_mesh_public_triangle_frag_spv[] = {
#include "g13_mesh_public_triangle.frag.spv.h"
};

static const uint32_t g13_mesh_public_triangle_task_spv[] = {
#include "g13_mesh_public_triangle.task.spv.h"
};

static const uint32_t g13_mesh_public_triangle_payload_spv[] = {
#include "g13_mesh_public_triangle_payload.mesh.spv.h"
};

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult result_ = (call);                                               \
      if (result_ != VK_SUCCESS) {                                             \
         fprintf(stderr, "%s: %d\n", #call, result_);                       \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static bool
has_extension(VkPhysicalDevice physical, const char *name)
{
   uint32_t count = 0;
   CHECK(vkEnumerateDeviceExtensionProperties(physical, NULL, &count, NULL));
   VkExtensionProperties *extensions = calloc(count, sizeof(*extensions));
   assert(extensions != NULL);
   CHECK(vkEnumerateDeviceExtensionProperties(physical, NULL, &count,
                                               extensions));

   bool found = false;
   for (uint32_t i = 0; i < count; ++i)
      found |= strcmp(extensions[i].extensionName, name) == 0;
   free(extensions);
   return found;
}

static VkPhysicalDevice
get_physical(VkInstance instance)
{
   uint32_t count = 1;
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   assert(count == 1 && physical != VK_NULL_HANDLE);
   return physical;
}

static void
check_negative(VkPhysicalDevice physical)
{
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &mesh_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);

   VkPhysicalDeviceMeshShaderPropertiesEXT mesh_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &mesh_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties);

   assert(!has_extension(physical, VK_EXT_MESH_SHADER_EXTENSION_NAME));
   assert(!mesh_features.taskShader && !mesh_features.meshShader &&
          !mesh_features.multiviewMeshShader &&
          !mesh_features.primitiveFragmentShadingRateMeshShader &&
          !mesh_features.meshShaderQueries);
   assert(mesh_properties.maxTaskWorkGroupTotalCount == 0);
   assert(mesh_properties.maxMeshWorkGroupTotalCount == 0);
   assert(mesh_properties.maxMeshWorkGroupInvocations == 0);
   assert(mesh_properties.maxMeshOutputVertices == 0);
   assert(mesh_properties.maxMeshOutputPrimitives == 0);
   assert(mesh_properties.maxMeshOutputComponents == 0);
   puts("mesh public API: extension/features/properties absent PASS");
}

static void
check_limits(const VkPhysicalDeviceMeshShaderPropertiesEXT *p)
{
   assert(p->maxTaskWorkGroupTotalCount == 4096);
   for (unsigned i = 0; i < 3; ++i)
      assert(p->maxTaskWorkGroupCount[i] == 4096);
   assert(p->maxTaskWorkGroupInvocations == 1024);
   assert(p->maxTaskWorkGroupSize[0] == 1024);
   assert(p->maxTaskWorkGroupSize[1] == 1024);
   assert(p->maxTaskWorkGroupSize[2] == 64);
   assert(p->maxTaskPayloadSize == 4096);
   assert(p->maxTaskSharedMemorySize == 28 * 1024);
   assert(p->maxTaskPayloadAndSharedMemorySize == 32 * 1024);
   assert(p->maxMeshWorkGroupTotalCount == 64);
   for (unsigned i = 0; i < 3; ++i) {
      assert(p->maxMeshWorkGroupCount[i] == 65535);
   }
   assert(p->maxMeshWorkGroupSize[0] == 1024);
   assert(p->maxMeshWorkGroupSize[1] == 1024);
   assert(p->maxMeshWorkGroupSize[2] == 64);
   assert(p->maxMeshWorkGroupInvocations == 1024);
   assert(p->maxMeshSharedMemorySize == 28 * 1024);
   assert(p->maxMeshPayloadAndSharedMemorySize == 28 * 1024);
   assert(p->maxMeshOutputMemorySize == 48 * 1024);
   assert(p->maxMeshPayloadAndOutputMemorySize == 48 * 1024);
   assert(p->maxMeshOutputComponents == 128);
   assert(p->maxMeshOutputVertices == 256);
   assert(p->maxMeshOutputPrimitives == 256);
   assert(p->maxMeshOutputLayers == 1);
   assert(p->maxMeshMultiviewViewCount == 1);
   assert(p->meshOutputPerVertexGranularity == 16);
   assert(p->meshOutputPerPrimitiveGranularity == 16);
   assert(p->maxPreferredTaskWorkGroupInvocations == 32);
   assert(p->maxPreferredMeshWorkGroupInvocations == 32);
}

static void
check_required_limit_gaps(
   const VkPhysicalDeviceMeshShaderPropertiesEXT *p)
{
   /* VK_EXT_mesh_shader Required_Limits. Keep this table executable so the
    * opt-in profile cannot be mistaken for conformant exposure. */
   assert(p->maxTaskWorkGroupTotalCount < (1u << 22));
   for (unsigned i = 0; i < 3; ++i)
      assert(p->maxTaskWorkGroupCount[i] < 65535);
   assert(p->maxTaskPayloadSize < 16384);
   assert(p->maxMeshWorkGroupTotalCount < (1u << 22));
   for (unsigned i = 0; i < 3; ++i)
      assert(p->maxMeshWorkGroupCount[i] >= 65535);
   assert(p->maxMeshWorkGroupInvocations >= 128);
   assert(p->maxMeshWorkGroupSize[0] >= 128);
   assert(p->maxMeshWorkGroupSize[1] >= 128);
   assert(p->maxMeshWorkGroupSize[2] < 128);
   assert(p->maxMeshSharedMemorySize >= 28672);
   assert(p->maxMeshPayloadAndSharedMemorySize >= 28672);
   assert(p->maxMeshOutputMemorySize >= 32768);
   assert(p->maxMeshPayloadAndOutputMemorySize >= 48128);
   assert(p->maxMeshOutputComponents >= 128);
   assert(p->maxMeshOutputVertices >= 256);
   assert(p->maxMeshOutputPrimitives >= 256);
   assert(p->maxMeshOutputLayers < 8);
   assert(p->meshOutputPerVertexGranularity <= 32);
   assert(p->meshOutputPerPrimitiveGranularity <= 32);
}

static VkDevice
create_device(VkPhysicalDevice physical, bool task)
{
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = true,
   };
   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .pNext = &features13,
      .taskShader = task,
      .meshShader = true,
   };
   const char *extensions[] = {VK_EXT_MESH_SHADER_EXTENSION_NAME};
   const VkDeviceCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &mesh_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device = VK_NULL_HANDLE;
   CHECK(vkCreateDevice(physical, &info, NULL, &device));
   return device;
}

static VkShaderModule
shader_module(VkDevice device, const uint32_t *code, size_t size)
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

static uint32_t
first_memory_type(VkPhysicalDevice physical, uint32_t bits)
{
   VkPhysicalDeviceMemoryProperties properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &properties);
   for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if (bits & (1u << i))
         return i;
   }
   assert(!"no compatible memory type");
   return 0;
}

static void
record_public_triangle(VkPhysicalDevice physical, VkDevice device, bool task)
{
   VkShaderModule mesh = shader_module(
      device,
      task ? g13_mesh_public_triangle_payload_spv
           : g13_mesh_public_triangle_spv,
      task ? sizeof(g13_mesh_public_triangle_payload_spv)
           : sizeof(g13_mesh_public_triangle_spv));
   VkShaderModule task_module = task
      ? shader_module(device, g13_mesh_public_triangle_task_spv,
                      sizeof(g13_mesh_public_triangle_task_spv))
      : VK_NULL_HANDLE;
   VkShaderModule fragment = shader_module(
      device, g13_mesh_public_triangle_frag_spv,
      sizeof(g13_mesh_public_triangle_frag_spv));

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));

   const VkViewport viewport = {0, 0, 64, 64, 0, 1};
   const VkRect2D scissor = {{0, 0}, {64, 64}};
   const VkPipelineRenderingCreateInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
   };
   VkPipelineShaderStageCreateInfo stages[3] = {0};
   uint32_t stage_count = 0;
   if (task) {
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
         .module = task_module,
         .pName = "main",
      };
   }
   stages[stage_count++] = (VkPipelineShaderStageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
         .module = mesh,
         .pName = "main",
      };
   stages[stage_count++] = (VkPipelineShaderStageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragment,
         .pName = "main",
      };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering_info,
      .stageCount = stage_count,
      .pStages = stages,
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
      },
      .layout = layout,
   };
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline));

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
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {64, 64}},
      .layerCount = 1,
   };
   vkCmdBeginRendering(command, &rendering);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

   PFN_vkCmdDrawMeshTasksEXT draw =
      (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(
         device, "vkCmdDrawMeshTasksEXT");
   assert(draw != NULL);
   draw(command, 0, 1, 1);
   draw(command, 2, 1, 1);

   VkBuffer indirect = VK_NULL_HANDLE;
   VkDeviceMemory indirect_memory = VK_NULL_HANDLE;
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(VkDrawMeshTasksIndirectCommandEXT),
      .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
   };
   CHECK(vkCreateBuffer(device, &buffer_info, NULL, &indirect));
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, indirect, &requirements);
   const VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = first_memory_type(
         physical, requirements.memoryTypeBits),
   };
   CHECK(vkAllocateMemory(device, &allocation, NULL, &indirect_memory));
   CHECK(vkBindBufferMemory(device, indirect, indirect_memory, 0));

   PFN_vkCmdDrawMeshTasksIndirectEXT draw_indirect =
      (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(
         device, "vkCmdDrawMeshTasksIndirectEXT");
   assert(draw_indirect != NULL);
   draw_indirect(command, indirect, 0, 1,
                 sizeof(VkDrawMeshTasksIndirectCommandEXT));
   vkCmdEndRendering(command);
   CHECK(vkEndCommandBuffer(command));

   vkDestroyCommandPool(device, pool, NULL);
   if (indirect != VK_NULL_HANDLE)
      vkDestroyBuffer(device, indirect, NULL);
   if (indirect_memory != VK_NULL_HANDLE)
      vkFreeMemory(device, indirect_memory, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, fragment, NULL);
   vkDestroyShaderModule(device, mesh, NULL);
   if (task_module != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, task_module, NULL);
}

int
main(int argc, char **argv)
{
   assert(argc == 2);
   const bool supported = strcmp(argv[1], "supported") == 0;
   assert(supported || strcmp(argv[1], "unsupported") == 0);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&info, NULL, &instance));
   VkPhysicalDevice physical = get_physical(instance);

   if (!supported) {
      check_negative(physical);
      vkDestroyInstance(instance, NULL);
      return 0;
   }

   VkPhysicalDeviceMeshShaderFeaturesEXT mesh_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &mesh_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   assert(has_extension(physical, VK_EXT_MESH_SHADER_EXTENSION_NAME));
   assert(mesh_features.meshShader && mesh_features.taskShader &&
          !mesh_features.multiviewMeshShader &&
          !mesh_features.primitiveFragmentShadingRateMeshShader &&
          !mesh_features.meshShaderQueries);

   VkPhysicalDeviceMeshShaderPropertiesEXT mesh_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &mesh_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties);
   check_limits(&mesh_properties);
   check_required_limit_gaps(&mesh_properties);

   VkDevice mesh_device = create_device(physical, false);
   record_public_triangle(physical, mesh_device, false);
   vkDestroyDevice(mesh_device, NULL);
   VkDevice task_device = create_device(physical, true);
   record_public_triangle(physical, task_device, true);
   vkDestroyDevice(task_device, NULL);
   vkDestroyInstance(instance, NULL);
   puts("G13 mesh public API: mesh/task feature/property, pipeline, direct "
        "and indirect record PASS");
   return 0;
}
