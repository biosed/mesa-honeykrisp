/* SPDX-License-Identifier: MIT
 * Queue selection and callback serialization without a DRM device.
 */
#include <stdio.h>
#include <stdlib.h>
#include "util/os_time.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

static struct hk_queue queues[6];
static int entered, released, finished, independent;

static void
await(int *flag)
{
   int64_t end = os_time_get_nano() + 2 * ONE_SECOND_IN_NS;
   while (!p_atomic_read(flag)) {
      CHECK(os_time_get_nano() < end);
      const struct timespec delay = {.tv_nsec = 1000000};
      thrd_sleep(&delay, NULL);
   }
}

static VkResult
driver_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   if (queue == &queues[2].vk) {
      p_atomic_set(&independent, 1);
      return VK_SUCCESS;
   }
   CHECK(queue == &queues[0].vk);
   p_atomic_inc(&entered);
   await(&released);
   p_atomic_inc(&finished);
   return VK_SUCCESS;
}

static void
submit(unsigned index)
{
   VkSubmitInfo2 info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
   CHECK(hk_QueueSubmit2(vk_queue_to_handle(&queues[index].vk), 1, &info,
                         VK_NULL_HANDLE) == VK_SUCCESS);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED;
   dev->vk.timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   list_inithead(&dev->vk.queues);
   dev->dev.chip = AGX_CHIP_G17C;

   for (unsigned i = 0; i < ARRAY_SIZE(queues); i++) {
      VkDeviceQueueCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .flags = i == 3 ? VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT : 0,
         .queueFamilyIndex = i % 3,
         .queueCount = 1};
      CHECK(vk_queue_init(&queues[i].vk, &dev->vk, &info, 0) == VK_SUCCESS);
      queues[i].drm.virt_ring_idx = i == 2 ? 1 : 2;
      queues[i].drm.id = 42 + i;
      queues[i].vk.driver_submit = driver_submit;
      if (i == 5)
         dev->dev.chip = AGX_CHIP_G17P;
      hk_queue_share(dev, &queues[i]);
   }
   dev->dev.chip = AGX_CHIP_G17C;
   CHECK(queues[0].vk.real_queue == &queues[0].vk);
   CHECK(queues[1].vk.real_queue == &queues[0].vk);
   CHECK(!queues[2].vk.real_queue);
   CHECK(!queues[3].vk.real_queue);
   CHECK(queues[4].vk.real_queue == &queues[0].vk);
   CHECK(!queues[5].vk.real_queue);
   CHECK(!queues[1].vk.driver_submit && !queues[4].vk.driver_submit);

   submit(0);
   await(&entered);
   submit(1);
   submit(2);
   await(&independent);
   const struct timespec delay = {.tv_nsec = 20000000};
   thrd_sleep(&delay, NULL);
   CHECK(p_atomic_read(&entered) == 1);
   p_atomic_set(&released, 1);

   /* Match device teardown: aliases must stop before the backing queue. */
   while (!list_is_empty(&dev->vk.queues)) {
      struct vk_queue *queue =
         list_last_entry(&dev->vk.queues, struct vk_queue, link);
      vk_queue_finish(queue);
   }
   CHECK(p_atomic_read(&entered) == 2);
   CHECK(p_atomic_read(&finished) == 2);
   vk_object_base_finish(&dev->vk.base);
   free(pdev);
   free(dev);
   puts(
      "queue matching, serialization, independent priority, reverse finish: PASS");
   return 0;
}
