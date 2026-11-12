/* SPDX-License-Identifier: MIT
 * Sparse mapping changes must follow completion, not merely submission, of
 * their input semaphores. Exercise the real bind path with an in-memory device.
 */
#include <stdio.h>
#include <stdlib.h>
#include "util/os_time.h"
#include "hk_buffer.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_queue.h"
#include "vk_sync.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

static int entered, complete, binds;
static bool wait_error, no_wait;

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
   return VK_SUCCESS;
}

static void
sync_finish(struct vk_device *dev, struct vk_sync *sync)
{
}

static VkResult
sync_signal(struct vk_device *dev, struct vk_sync *sync, uint64_t value)
{
   return VK_SUCCESS;
}

static VkResult
sync_get_value(struct vk_device *dev, struct vk_sync *sync, uint64_t *value)
{
   *value = p_atomic_read(&complete) ? 7 : 0;
   return VK_SUCCESS;
}

static VkResult
sync_wait(struct vk_device *dev, struct vk_sync *sync, uint64_t value,
          enum vk_sync_wait_flags flags, uint64_t timeout)
{
   /* The signal operation is already submitted, but has not completed. */
   if (flags & VK_SYNC_WAIT_PENDING)
      return VK_SUCCESS;

   CHECK(value == 7);
   p_atomic_set(&entered, 1);
   if (wait_error)
      return VK_ERROR_UNKNOWN;

   while (!p_atomic_read(&complete)) {
      if ((uint64_t)os_time_get_nano() >= timeout)
         return VK_TIMEOUT;
      const struct timespec delay = {.tv_nsec = 1000000};
      thrd_sleep(&delay, NULL);
   }
   return VK_SUCCESS;
}

static const struct vk_sync_type sync_type = {
   .size = sizeof(struct vk_sync),
   .features = VK_SYNC_FEATURE_TIMELINE | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = sync_init,
   .finish = sync_finish,
   .signal = sync_signal,
   .get_value = sync_get_value,
   .wait = sync_wait,
};

static int
bo_bind(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
        uint32_t count)
{
   CHECK(no_wait || p_atomic_read(&complete));
   CHECK(count == 2);
   CHECK(ops[0].handle == 42 && ops[1].handle == 42);
   CHECK(ops[0].addr == 0x10000 && ops[0].range == 0x4000);
   CHECK(ops[1].addr == ops[0].addr + dev->sparse_ro_offset);
   p_atomic_inc(&binds);
   return 0;
}

struct fixture {
   struct hk_device *dev;
   struct vk_queue_submit submit;
   VkResult result;
};

static int
submit(void *data)
{
   struct fixture *f = data;
   f->result = hk_queue_submit_bind_sparse_memory(f->dev, &f->submit);
   return 0;
}

static void
run(unsigned mode)
{
   struct fixture f = {.dev = calloc(1, sizeof(struct hk_device))};
   CHECK(f.dev);
   struct vk_device *dev = &f.dev->vk;
   vk_object_base_init(dev, &dev->base, VK_OBJECT_TYPE_DEVICE);
   dev->_lost.reported = true;
   /* G17P's GPU can poll for a host vkSetEvent, so binds there must not
    * wait for their inputs on the submitting thread. */
   no_wait = mode == 4;
   f.dev->dev.chip = no_wait ? AGX_CHIP_G17P : AGX_CHIP_G17C;
   f.dev->dev.ops.bo_bind = bo_bind;
   f.dev->dev.sparse_ro_offset = 1ull << 38;
   struct agx_va va = {.addr = 0x10000};
   struct hk_buffer buffer = {.va = &va};
   struct agx_bo bo = {.uapi_handle = 42};
   struct hk_device_memory memory = {.bo = &bo};
   vk_object_base_init(dev, &buffer.vk.base, VK_OBJECT_TYPE_BUFFER);
   vk_object_base_init(dev, &memory.vk.base, VK_OBJECT_TYPE_DEVICE_MEMORY);
   VkSparseMemoryBind bind = {.size = 0x4000,
                              .memory = hk_device_memory_to_handle(&memory)};
   VkSparseBufferMemoryBindInfo info = {.buffer = hk_buffer_to_handle(&buffer),
                                        .bindCount = 1,
                                        .pBinds = &bind};
   struct vk_sync sync;
   CHECK(vk_sync_init(dev, &sync, &sync_type, VK_SYNC_IS_TIMELINE, 0) ==
         VK_SUCCESS);
   struct vk_sync_wait wait = {.sync = &sync, .wait_value = 7};
   f.submit.wait_count = 1;
   f.submit.waits = &wait;
   f.submit.buffer_bind_count = 1;
   f.submit.buffer_binds = &info;
   entered = binds = 0;
   complete = mode == 3;
   wait_error = mode == 2;
   CHECK(vk_sync_wait(dev, &sync, 7, VK_SYNC_WAIT_PENDING, 0) == VK_SUCCESS);
   thrd_t thread;
   CHECK(thrd_create(&thread, submit, &f) == thrd_success);
   if (!no_wait)
      await(&entered);
   if (mode < 2) {
      const struct timespec delay = {.tv_nsec = 20000000};
      thrd_sleep(&delay, NULL);
      CHECK(!p_atomic_read(&binds));
      if (mode == 0)
         p_atomic_set(&complete, 1);
      else
         p_atomic_set(&dev->_lost.lost, 1);
   }
   CHECK(thrd_join(thread, NULL) == thrd_success);
   CHECK(f.result == (mode == 1   ? VK_ERROR_DEVICE_LOST
                      : mode == 2 ? VK_ERROR_UNKNOWN
                                  : VK_SUCCESS));
   CHECK(p_atomic_read(&binds) == (mode == 0 || mode >= 3));
   CHECK(!no_wait || !p_atomic_read(&entered));
   vk_sync_finish(dev, &sync);
   vk_object_base_finish(&memory.vk.base);
   vk_object_base_finish(&buffer.vk.base);
   vk_object_base_finish(&dev->base);
   free(f.dev);
   printf("sparse wait mode=%u: PASS\n", mode);
}

int
main(void)
{
   for (unsigned mode = 0; mode < 5; mode++)
      run(mode);
   return 0;
}
