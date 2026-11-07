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

#include "g17p_wsi_triangle_frag.h"

static const uint32_t g13_mesh_public_triangle_spv[] = {
#include "g13_mesh_public_triangle.spv.h"
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
   assert(p->maxTaskWorkGroupTotalCount == 0);
   assert(p->maxTaskWorkGroupInvocations == 0);
   assert(p->maxTaskPayloadSize == 0);
   assert(p->maxMeshWorkGroupTotalCount == 4096);
   for (unsigned i = 0; i < 3; ++i) {
      assert(p->maxMeshWorkGroupCount[i] == 65535);
   }
   assert(p->maxMeshWorkGroupSize[0] == 1024);
   assert(p->maxMeshWorkGroupSize[1] == 1024);
   assert(p->maxMeshWorkGroupSize[2] == 64);
   assert(p->maxMeshWorkGroupInvocations == 1024);
   assert(p->maxMeshSharedMemorySize == 24 * 1024);
   assert(p->maxMeshPayloadAndSharedMemorySize == 24 * 1024);
   assert(p->maxMeshOutputMemorySize == 8192);
   assert(p->maxMeshPayloadAndOutputMemorySize == 8192);
   assert(p->maxMeshOutputComponents == 4);
   assert(p->maxMeshOutputVertices == 256);
   assert(p->maxMeshOutputPrimitives == 256);
   assert(p->maxMeshOutputLayers == 1);
   assert(p->maxMeshMultiviewViewCount == 1);
   assert(p->meshOutputPerVertexGranularity == 16);
   assert(p->meshOutputPerPrimitiveGranularity == 4);
   assert(p->maxPreferredTaskWorkGroupInvocations == 0);
   assert(p->maxPreferredMeshWorkGroupInvocations == 32);
}

static void
check_required_limit_gaps(
   const VkPhysicalDeviceMeshShaderPropertiesEXT *p)
{
   /* VK_EXT_mesh_shader Required_Limits. Keep this table executable so the
    * opt-in profile cannot be mistaken for conformant exposure while the
    * position-only prototype still falls below it. */
   assert(p->maxMeshWorkGroupTotalCount < (1u << 22));
   for (unsigned i = 0; i < 3; ++i)
      assert(p->maxMeshWorkGroupCount[i] >= 65535);
   assert(p->maxMeshWorkGroupInvocations >= 128);
   assert(p->maxMeshWorkGroupSize[0] >= 128);
   assert(p->maxMeshWorkGroupSize[1] >= 128);
   assert(p->maxMeshWorkGroupSize[2] < 128);
   assert(p->maxMeshSharedMemorySize < 28672);
   assert(p->maxMeshPayloadAndSharedMemorySize < 28672);
   assert(p->maxMeshOutputMemorySize < 32768);
   assert(p->maxMeshPayloadAndOutputMemorySize < 48128);
   assert(p->maxMeshOutputComponents < 128);
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
   VkResult result = vkCreateDevice(physical, &info, NULL, &device);
   if (task) {
      assert(result == VK_ERROR_FEATURE_NOT_PRESENT);
      assert(device == VK_NULL_HANDLE);
      return VK_NULL_HANDLE;
   }
   CHECK(result);
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

static void
record_public_triangle(VkDevice device)
{
   VkShaderModule mesh = shader_module(
      device, g13_mesh_public_triangle_spv,
      sizeof(g13_mesh_public_triangle_spv));
   VkShaderModule fragment = shader_module(
      device, g17p_wsi_triangle_frag_spv,
      sizeof(g17p_wsi_triangle_frag_spv));

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
   const VkPipelineShaderStageCreateInfo stages[] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
         .module = mesh,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragment,
         .pName = "main",
      },
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering_info,
      .stageCount = 2,
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
   vkCmdEndRendering(command);
   CHECK(vkEndCommandBuffer(command));

   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, fragment, NULL);
   vkDestroyShaderModule(device, mesh, NULL);
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
   assert(mesh_features.meshShader && !mesh_features.taskShader &&
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

   (void)create_device(physical, true);
   VkDevice device = create_device(physical, false);
   record_public_triangle(device);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   puts("G13 mesh public API: feature/property/pipeline/direct record PASS");
   return 0;
}
