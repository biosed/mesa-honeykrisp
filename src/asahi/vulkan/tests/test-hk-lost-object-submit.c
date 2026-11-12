/* SPDX-License-Identifier: MIT
 * Exercise real object teardown against a paused host submission. No GPU used.
 */
#include <stdio.h>
#include <stdlib.h>
#include "util/os_time.h"
#include "hk_buffer.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_fence.h"
#include "vk_semaphore.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)
static int entered, released, cleanup_started, cleanup_done, calls;
static bool pending_wait;
struct test_sync {
   struct vk_sync base;
   unsigned marker;
};
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
sync_init(struct vk_device *dev, struct vk_sync *sync, uint64_t value)
{
   ((struct test_sync *)sync)->marker = 1234;
   return VK_SUCCESS;
}
static void
sync_finish(struct vk_device *dev, struct vk_sync *sync)
{
}
static VkResult
sync_signal(struct vk_device *dev, struct vk_sync *sync, uint64_t value)
{
   CHECK(((struct test_sync *)sync)->marker == 1234);
   return VK_SUCCESS;
}
static VkResult
sync_get_value(struct vk_device *dev, struct vk_sync *sync, uint64_t *value)
{
   CHECK(((struct test_sync *)sync)->marker == 1234);
   *value = 0;
   return VK_SUCCESS;
}
static VkResult
sync_wait(struct vk_device *dev, struct vk_sync *sync, uint64_t value,
          enum vk_sync_wait_flags flags, uint64_t timeout)
{
   CHECK(flags & VK_SYNC_WAIT_PENDING);
   CHECK(pending_wait);
   p_atomic_set(&entered, 1);
   await(&released);
   CHECK(((struct test_sync *)sync)->marker == 1234);
   return VK_ERROR_DEVICE_LOST;
}
static const struct vk_sync_type sync_type = {
   .size = sizeof(struct test_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE |
               VK_SYNC_FEATURE_GPU_WAIT | VK_SYNC_FEATURE_GPU_MULTI_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = sync_init,
   .finish = sync_finish,
   .signal = sync_signal,
   .get_value = sync_get_value,
   .wait = sync_wait,
};
static VkResult
driver_submit(struct vk_queue *q, struct vk_queue_submit *s)
{
   p_atomic_inc(&calls);
   p_atomic_set(&entered, 1);
   await(&released);
   CHECK(s->buffer_bind_count == 1 && s->image_opaque_bind_count == 1);
   VK_FROM_HANDLE(hk_buffer, buffer, s->buffer_binds[0].buffer);
   VK_FROM_HANDLE(hk_image, image, s->image_opaque_binds[0].image);
   VK_FROM_HANDLE(hk_device_memory, mem, s->buffer_binds[0].pBinds[0].memory);
   CHECK(buffer->vk.size == 4096 && image->vk.extent.width == 64);
   CHECK(mem->bo->size == 4096 && mem->bo->refcnt == 2);
   CHECK(s->signal_count == 2);
   for (unsigned i = 0; i < s->signal_count; i++)
      CHECK(vk_sync_signal(q->base.device, s->signals[i].sync,
                           s->signals[i].signal_value) == VK_SUCCESS);
   return VK_SUCCESS;
}
struct fixture {
   struct hk_device *dev;
   struct hk_buffer *buffer;
   struct hk_image *image;
   struct hk_device_memory *memory;
   struct vk_semaphore *semaphore;
   struct vk_fence *fence;
   unsigned kind;
};
static void
destroy(struct fixture *f, unsigned kind)
{
   VkDevice dev = hk_device_to_handle(f->dev);
   switch (kind) {
   case 0:
      hk_DestroyBuffer(dev, hk_buffer_to_handle(f->buffer), NULL);
      f->buffer = NULL;
      break;
   case 1:
      hk_DestroyImage(dev, hk_image_to_handle(f->image), NULL);
      f->image = NULL;
      break;
   case 2:
      hk_FreeMemory(dev, hk_device_memory_to_handle(f->memory), NULL);
      f->memory = NULL;
      break;
   case 3:
      hk_DestroySemaphore(dev, vk_semaphore_to_handle(f->semaphore), NULL);
      f->semaphore = NULL;
      break;
   case 4:
      hk_DestroyFence(dev, vk_fence_to_handle(f->fence), NULL);
      f->fence = NULL;
      break;
   default:
      abort();
   }
}
static int
cleanup(void *data)
{
   struct fixture *f = data;
   p_atomic_set(&cleanup_started, 1);
   destroy(f, f->kind);
   p_atomic_set(&cleanup_done, 1);
   return 0;
}
static void
run(unsigned kind, bool wait)
{
   struct fixture f = {.kind = kind};
   f.dev = calloc(1, sizeof(*f.dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(f.dev && pdev);
   struct vk_device *dev = &f.dev->vk;
   vk_object_base_init(dev, &dev->base, VK_OBJECT_TYPE_DEVICE);
   dev->alloc = *vk_default_allocator();
   dev->physical = &pdev->vk;
   dev->_lost.reported = true;
   dev->submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED;
   dev->timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   list_inithead(&dev->queues);
   f.dev->dev.chip = AGX_CHIP_G17C;
   uint64_t used = 4096;
   pdev->mem_heaps[0].used = &used;
   struct agx_bo bo = {.size = 4096, .refcnt = 2};
   f.buffer = calloc(1, sizeof(*f.buffer));
   f.image = calloc(1, sizeof(*f.image));
   f.memory = calloc(1, sizeof(*f.memory));
   f.semaphore =
      calloc(1, offsetof(struct vk_semaphore, permanent) + sync_type.size);
   f.fence = calloc(1, offsetof(struct vk_fence, permanent) + sync_type.size);
   CHECK(f.buffer && f.image && f.memory && f.semaphore && f.fence);
   vk_object_base_init(dev, &f.buffer->vk.base, VK_OBJECT_TYPE_BUFFER);
   vk_object_base_init(dev, &f.image->vk.base, VK_OBJECT_TYPE_IMAGE);
   vk_object_base_init(dev, &f.memory->vk.base, VK_OBJECT_TYPE_DEVICE_MEMORY);
   vk_object_base_init(dev, &f.semaphore->base, VK_OBJECT_TYPE_SEMAPHORE);
   vk_object_base_init(dev, &f.fence->base, VK_OBJECT_TYPE_FENCE);
   f.buffer->vk.size = 4096;
   f.image->vk.extent.width = 64;
   f.memory->bo = &bo;
   f.semaphore->type = VK_SEMAPHORE_TYPE_TIMELINE;
   CHECK(vk_sync_init(dev, &f.semaphore->permanent, &sync_type,
                      VK_SYNC_IS_TIMELINE, 0) == VK_SUCCESS);
   CHECK(vk_sync_init(dev, &f.fence->permanent, &sync_type, 0, 0) ==
         VK_SUCCESS);
   struct hk_queue q = {0};
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1};
   CHECK(vk_queue_init(&q.vk, dev, &qi, 0) == VK_SUCCESS);
   q.vk.driver_submit = driver_submit;
   VkSparseMemoryBind binding = {
      .size = 4096,
      .memory = hk_device_memory_to_handle(f.memory)};
   VkSparseBufferMemoryBindInfo buffer = {
      .buffer = hk_buffer_to_handle(f.buffer),
      .bindCount = 1,
      .pBinds = &binding};
   VkSparseImageOpaqueMemoryBindInfo image = {
      .image = hk_image_to_handle(f.image),
      .bindCount = 1,
      .pBinds = &binding};
   VkSemaphore sem = vk_semaphore_to_handle(f.semaphore);
   uint64_t value = 2, wait_value = 1;
   VkTimelineSemaphoreSubmitInfo timeline = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &value,
      .waitSemaphoreValueCount = wait,
      .pWaitSemaphoreValues = &wait_value};
   VkBindSparseInfo bi = {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
                          .pNext = &timeline,
                          .bufferBindCount = 1,
                          .pBufferBinds = &buffer,
                          .imageOpaqueBindCount = 1,
                          .pImageOpaqueBinds = &image,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &sem,
                          .waitSemaphoreCount = wait,
                          .pWaitSemaphores = &sem};
   entered = released = cleanup_started = cleanup_done = calls = 0;
   pending_wait = wait;
   CHECK(vk_common_QueueBindSparse(vk_queue_to_handle(&q.vk), 1, &bi,
                                   vk_fence_to_handle(f.fence)) == VK_SUCCESS);
   await(&entered);
   p_atomic_set(&dev->_lost.lost, 1);
   thrd_t thread;
   CHECK(thrd_create(&thread, cleanup, &f) == thrd_success);
   await(&cleanup_started);
   const struct timespec delay = {.tv_nsec = 100000000};
   thrd_sleep(&delay, NULL);
   CHECK(!p_atomic_read(&cleanup_done));
   p_atomic_set(&released, 1);
   await(&cleanup_done);
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(p_atomic_read(&calls) == !wait);
   vk_queue_finish(&q.vk);
   for (unsigned other = 0; other < 5; other++)
      if (other != kind)
         destroy(&f, other);
   CHECK(bo.refcnt == 1 && used == 0);
   vk_object_base_finish(&dev->base);
   free(f.dev);
   free(pdev);
   printf("object=%u pending_wait=%u: PASS\n", kind, wait);
}
int
main(void)
{
   for (unsigned kind = 0; kind < 5; kind++)
      run(kind, false);
   run(3, true);
   return 0;
}
