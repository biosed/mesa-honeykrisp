/* SPDX-License-Identifier: MIT */
#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_queue.h"
#include "util/os_time.h"

#include <stdio.h>
#include <stdlib.h>

static int entered, release_worker, cleanup_started, cleanup_done, calls;
static void
require(bool condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "%s\n", message);
      exit(EXIT_FAILURE);
   }
}

static void
wait_for(int *flag)
{
   int64_t deadline = os_time_get_nano() + 2 * ONE_SECOND_IN_NS;
   while (!p_atomic_read(flag)) {
      require(os_time_get_nano() < deadline, "host thread did not make progress");
      const struct timespec delay = {.tv_nsec = 1000000};
      thrd_sleep(&delay, NULL);
   }
}

static VkResult
blocked_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   p_atomic_inc(&calls);
   p_atomic_set(&entered, 1);
   wait_for(&release_worker);
   /* The recording must remain readable through the whole callback, even
    * if another queue has reported device loss while this callback ran. */
   require(submit->command_buffer_count == 1, "missing command buffer");
   require(submit->command_buffers[0]->record_result == VK_SUCCESS,
           "recording was reset before host submission stopped");
   return VK_SUCCESS;
}

struct cleanup {
   struct vk_queue *queue;
   struct vk_command_buffer *cmd;
};

static int
cleanup_thread(void *data)
{
   struct cleanup *cleanup = data;
   p_atomic_set(&cleanup_started, 1);
   vk_queue_wait_lost_submissions(cleanup->queue);
   cleanup->cmd->record_result = VK_ERROR_UNKNOWN;
   vk_object_base_finish(&cleanup->cmd->base);
   free(cleanup->cmd);
   p_atomic_set(&cleanup_done, 1);
   return 0;
}

static void
run(bool deferred)
{
   struct vk_device device = {
      .alloc = *vk_default_allocator(),
      .submit_mode = deferred ? VK_QUEUE_SUBMIT_MODE_DEFERRED :
                                VK_QUEUE_SUBMIT_MODE_THREADED,
      .timeline_mode = VK_DEVICE_TIMELINE_MODE_NONE,
   };
   vk_object_base_init(NULL, &device.base, VK_OBJECT_TYPE_DEVICE);
   device._lost.reported = true;
   list_inithead(&device.queues);
   struct vk_queue queue;
   const VkDeviceQueueCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
   };
   require(vk_queue_init(&queue, &device, &create, 0) == VK_SUCCESS,
           "queue initialization failed");
   queue.driver_submit = blocked_submit;
   p_atomic_set(&calls, 0);
   p_atomic_set(&entered, 0);
   p_atomic_set(&release_worker, 0);
   p_atomic_set(&cleanup_started, 0);
   p_atomic_set(&cleanup_done, 0);

   struct vk_command_pool pool = {.queue_family_index = 0};
   struct vk_command_buffer *cmd = calloc(1, sizeof(*cmd));
   require(cmd != NULL, "allocation failed");
   vk_object_base_init(&device, &cmd->base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->pool = &pool;
   cmd->state = MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE;
   cmd->record_result = VK_SUCCESS;

   if (deferred) {
      /* Deferred records have no worker to drain them after loss. Exercise
       * their cancellation and cleanup, including an empty second record. */
      for (unsigned i = 0; i < 2; i++) {
         struct vk_queue_submit *s = vk_zalloc(&device.alloc, sizeof(*s), 8,
                                               VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
         require(s != NULL, "submission allocation failed");
         list_addtail(&s->link, &queue.submit.submits);
      }
   } else {
      const VkCommandBufferSubmitInfo cb = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
         .commandBuffer = vk_command_buffer_to_handle(cmd),
      };
      const VkSubmitInfo2 submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
         .commandBufferInfoCount = 1,
         .pCommandBufferInfos = &cb,
      };
      require(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &submit,
                                      VK_NULL_HANDLE) == VK_SUCCESS,
              "first submission failed");
      wait_for(&entered);
      require(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &submit,
                                      VK_NULL_HANDLE) == VK_SUCCESS,
              "second submission failed");
   }

   /* Model another queue publishing device loss, without a physical device. */
   p_atomic_set(&device._lost.lost, 1);
   struct cleanup cleanup = {&queue, cmd};
   thrd_t thread;
   require(thrd_create(&thread, cleanup_thread, &cleanup) == thrd_success,
           "cleanup thread creation failed");
   wait_for(&cleanup_started);
   if (!deferred) {
      const struct timespec delay = {.tv_nsec = 100000000};
      thrd_sleep(&delay, NULL);
      require(!p_atomic_read(&cleanup_done),
              "cleanup returned while the submit callback still held recording metadata");
      p_atomic_set(&release_worker, 1);
   }
   wait_for(&cleanup_done);
   require(thrd_join(thread, NULL) == thrd_success, "cleanup join failed");
   require(list_is_empty(&queue.submit.submits), "pending records survived loss");
   require(p_atomic_read(&calls) == (deferred ? 0 : 1),
           "a queued callback ran after device loss");
   vk_queue_finish(&queue);
   vk_object_base_finish(&device.base);
   printf("%s: PASS\n", deferred ? "deferred cancellation" :
          "active callback, queued cancellation, recording teardown");
}

int
main(void)
{
   run(false);
   run(true);
   return 0;
}
