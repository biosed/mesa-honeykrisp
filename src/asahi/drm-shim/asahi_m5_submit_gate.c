/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi_m5_submit_gate.h"

#include <errno.h>
#include <stdbool.h>
#include <string.h>

#include "drm-uapi/asahi_drm.h"
#include "util/macros.h"

#define REFUSE(err, why)                                                       \
   do {                                                                        \
      *reason = (why);                                                         \
      return -(err);                                                           \
   } while (0)

/* g17/command.rs */
#define M5_MAX_TILE_BUFFER     65536u
#define MAX_DIMENSION          16384u
#define MAX_LAYERS             2048u
#define MAX_COMMANDS           64u
#define PAGE_SIZE              0x4000u
#define M5_RENDER_FLAGS                                                        \
   (DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES | DRM_ASAHI_RENDER_RSRC_SPEC_HI |     \
    DRM_ASAHI_RENDER_DBIAS_IS_INT | DRM_ASAHI_RENDER_FEW_PRIMITIVES |          \
    DRM_ASAHI_RENDER_FRAGMENT_BARRIERS | DRM_ASAHI_RENDER_SHADER_WRITES |      \
    DRM_ASAHI_RENDER_STAGE_PREFIXES | DRM_ASAHI_RENDER_WHOLE_ATTACHMENTS)
#define M5_COMPUTE_FLAGS                                                       \
   (DRM_ASAHI_COMPUTE_USES_MATRIX | DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS)

static uint64_t
pages(uint64_t value, uint64_t size)
{
   return (value + size - 1) / size;
}

/* The per-entry limits of g17/fw/m5/render_demand.rs: the large-page counts
 * each program contributes must fit RenderDemand::resource_flags(). */
static bool
fragment_storage_fits(uint32_t primary, uint32_t secondary, uint32_t linked,
                      uint32_t minimum)
{
   uint64_t frame = (uint64_t)primary + linked;
   if (frame < minimum)
      frame = minimum;
   frame = frame * 32 + 0x4380;
   return pages(frame, 0x10000) <= 0xff &&
          pages((uint64_t)secondary * 32, 0x10000) <= 0xf;
}

static bool
tile_storage_fits(uint32_t threadgroup, uint32_t imageblock,
                  uint32_t multisample)
{
   return pages(threadgroup, 0x10000) <= 3 &&
          pages(imageblock ? (uint64_t)imageblock + 64 : 0, 0x10000) <= 0xff &&
          pages(multisample, 0x10000) <= 0xff;
}

/* Programs::validate */
static bool
render_program_valid(const struct drm_asahi_render_program *p)
{
   if (p->stage > 1 || p->flags != 0)
      return false;
   if (p->stage == 0 &&
       (p->auxiliary_count || p->threadgroup_bytes || p->imageblock_bytes ||
        p->multisample_bytes))
      return false;
   return fragment_storage_fits(p->primary_bytes, p->secondary_bytes,
                                p->linked_bytes, p->minimum_primary_bytes) &&
          (p->stage == 0 ||
           tile_storage_fits(p->threadgroup_bytes, p->imageblock_bytes,
                             p->multisample_bytes));
}

/* Programs::parse and Programs::parse_v2, for the forms Mesa's Gallium and
 * Vulkan drivers send without mesh, object or ray-query stages. */
static int
parse_render_programs(const uint8_t *bytes, size_t size, bool v2,
                      struct asahi_m5_gate_stats *stats, const char **reason)
{
   const size_t stride = v2 ? sizeof(struct drm_asahi_render_program_v2)
                            : sizeof(struct drm_asahi_render_program);
   if (size % stride)
      REFUSE(EINVAL, "render program list is not a whole number of records");

   for (size_t at = 0; at < size; at += stride) {
      struct drm_asahi_render_program_v2 entry = {0};
      memcpy(&entry, bytes + at, stride);
      if (v2) {
         const bool mesh = entry.usage_flags & (DRM_ASAHI_RENDER_PROGRAM_MESH |
                                                DRM_ASAHI_RENDER_PROGRAM_OBJECT);
         if (entry.reserved || (entry.usage_flags & ~0xfu) ||
             (entry.usage_flags & 6) == 6 || (!mesh && entry.threads))
            REFUSE(EINVAL, "render program v2 usage");
         if (mesh || entry.scs_layout || entry.auxiliary_allocation ||
             (entry.usage_flags & DRM_ASAHI_RENDER_PROGRAM_SCS) ||
             entry.intersection_pool || entry.intersection_count)
            continue; /* not modelled */
      }
      if (!render_program_valid(&entry.legacy))
         REFUSE(EINVAL, "render program record");
      stats->render_programs++;
   }
   return 0;
}

/* ComputePrograms::parse, with the per-dispatch limits of
 * RenderDemand::include_compute (96 resident warps, 112 KiB shared). */
static int
parse_compute_programs(const uint8_t *bytes, size_t size,
                       struct asahi_m5_gate_stats *stats, const char **reason)
{
   const size_t stride = sizeof(struct drm_asahi_compute_program);
   if (size % stride)
      REFUSE(EINVAL, "compute program list is not a whole number of records");

   for (size_t at = 0; at < size; at += stride) {
      struct drm_asahi_compute_program p;
      memcpy(&p, bytes + at, stride);
      if (p.flags || !p.threads)
         REFUSE(EINVAL, "compute program record");

      const uint64_t warps = pages(p.threads, 32);
      uint64_t groups = 96 / warps;
      if (p.threadgroup_bytes) {
         const uint64_t aligned = pages(p.threadgroup_bytes, 64) * 64;
         groups = MIN2(groups, 114688 / aligned);
      }
      if (!groups ||
          !fragment_storage_fits(p.primary_bytes, p.secondary_bytes,
                                 p.linked_bytes, p.minimum_primary_bytes) ||
          pages(p.threadgroup_bytes, 0x10000) > 3)
         REFUSE(EINVAL, "compute program demand");
      stats->compute_programs++;
   }
   return 0;
}

/* surface() */
static bool
zls_valid(const struct drm_asahi_zls_buffer *zls, unsigned layers)
{
   if ((zls->base == 0 &&
        (zls->comp_base || zls->stride || zls->comp_stride)) ||
       (zls->comp_base == 0 && zls->comp_stride) ||
       (layers > 1 && zls->base && !zls->stride) ||
       (zls->stride && (zls->stride & (PAGE_SIZE - 1)) != 1) ||
       (zls->comp_stride & (PAGE_SIZE - 1)))
      return false;
   return true;
}

static bool
helper_empty(const struct drm_asahi_helper_program *h)
{
   return !h->binary && !h->cfg && !h->data;
}

static bool
utile_valid(unsigned w, unsigned h)
{
   return (w == 16 && h == 16) || (w == 32 && h == 16) || (w == 32 && h == 32);
}

#define Z_LOAD           (1ull << 15)
#define S_LOAD           (1ull << 14)
#define Z_STORE          (1ull << 19)
#define S_STORE          (1ull << 18)
#define Z_LOAD_COMPRESS  (1ull << 2)
#define S_LOAD_COMPRESS  (1ull << 4)
#define Z_STORE_COMPRESS (1ull << 6)
#define S_STORE_COMPRESS (1ull << 8)
#define Z_FORMAT         (3ull << 25)

/* compressed_zls_form() */
static bool
compressed_zls_form(const struct drm_asahi_cmd_render *c)
{
   const uint64_t known = Z_LOAD | S_LOAD | Z_STORE | S_STORE |
                          Z_LOAD_COMPRESS | S_LOAD_COMPRESS | Z_STORE_COMPRESS |
                          S_STORE_COMPRESS | Z_FORMAT;
   const uint64_t z = c->zls_ctrl;
   const bool zc = z & (Z_LOAD_COMPRESS | Z_STORE_COMPRESS);
   const bool sc = z & (S_LOAD_COMPRESS | S_STORE_COMPRESS);
   return z && !(z & ~known) &&
          (!(z & Z_LOAD) || (z & Z_LOAD_COMPRESS)) &&
          (!(z & S_LOAD) || (z & S_LOAD_COMPRESS)) &&
          (!(z & Z_STORE) || (z & Z_STORE_COMPRESS)) &&
          (!(z & S_STORE) || (z & S_STORE_COMPRESS)) &&
          (!zc || (c->depth.base && c->depth.comp_base)) &&
          (!sc || (c->stencil.base && c->stencil.comp_base)) &&
          (zc || !c->depth.comp_base) && (sc || !c->stencil.comp_base);
}

/* single_layer_compressed_depth_zls() */
static bool
single_layer_compressed_depth_zls(const struct drm_asahi_cmd_render *c,
                                  unsigned samples_log2)
{
   const uint64_t zcomp = Z_LOAD_COMPRESS | Z_STORE_COMPRESS;
   const uint64_t known = Z_LOAD | S_LOAD | Z_STORE | S_STORE | zcomp;
   return c->layers == 1 && (samples_log2 == 1 || samples_log2 == 2) &&
          !(c->zls_ctrl & ~known) && (c->zls_ctrl & zcomp) == zcomp &&
          c->depth.base && c->depth.comp_base && !c->stencil.comp_base &&
          (!(c->zls_ctrl & (S_LOAD | S_STORE)) || c->stencil.base);
}

/* m5_render_submission::validate. The compressed single-layer multisample
 * depth/stencil forms are admitted when the kernel reports
 * DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION (its m5_depth_msaa_compression
 * opt-in). */
static int
m5_render_admission(const struct drm_asahi_cmd_render *c, unsigned samples_log2,
                    uint64_t features, const char **reason)
{
   const bool msaa_compression =
      features & DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION;
   const uint32_t tile = (uint32_t)c->utile_width_px * c->utile_height_px;
   const bool single = samples_log2 == 0 && c->sample_size_B % 8 == 0 &&
                       c->sample_size_B * tile <= 65536 &&
                       (c->sample_size_B != 0 || c->layers == 1);
   const bool plain = c->zls_ctrl == 0 ||
                      ((c->zls_ctrl & 0x154) == 0 && !c->depth.comp_base &&
                       !c->stencil.comp_base);
   const bool form =
      plain || (c->layers > 1 && compressed_zls_form(c)) ||
      (msaa_compression &&
       (single_layer_compressed_depth_zls(c, samples_log2) ||
        (c->layers == 1 && (samples_log2 == 1 || samples_log2 == 2) &&
         compressed_zls_form(c))));
   const bool multi = form && samples_log2 >= 1 && samples_log2 <= 3 &&
                      c->sample_size_B >= 8 && c->sample_size_B % 8 == 0 &&
                      (tile << samples_log2) <= 2048 &&
                      ((c->sample_size_B * tile) << samples_log2) <= 65536;
   if (!(single || multi))
      REFUSE(EOPNOTSUPP, "M5 render admission (tile, samples or ZLS form)");
   if (c->zls_ctrl && !c->depth.base && !c->stencil.base)
      REFUSE(EOPNOTSUPP, "M5 render admission (ZLS control without storage)");
   return 0;
}

static int
validate_render(const uint8_t *bytes, size_t size, bool have_resources,
                uint64_t features, const char **reason)
{
   struct drm_asahi_cmd_render c = {0};
   memcpy(&c, bytes, MIN2(size, sizeof(c)));

   /* render_payload: a tail past the render command is a stage-prefix
    * extension and must be zero unless the render is staged. */
   if (!(c.flags & DRM_ASAHI_RENDER_STAGE_PREFIXES)) {
      for (size_t i = sizeof(c); i < size; ++i) {
         if (bytes[i])
            REFUSE(EINVAL, "render stage-prefix tail without the flag");
      }
   }

   if ((c.flags & ~M5_RENDER_FLAGS) || c.width_px < 1 ||
       c.width_px > MAX_DIMENSION || c.height_px < 1 ||
       c.height_px > MAX_DIMENSION || c.layers < 1 || c.layers > MAX_LAYERS ||
       !utile_valid(c.utile_width_px, c.utile_height_px))
      REFUSE(EINVAL, "render flags, dimensions, layers or utile");

   unsigned samples_log2;
   switch (c.samples) {
   case 1: samples_log2 = 0; break;
   case 2: samples_log2 = 1; break;
   case 4: samples_log2 = 2; break;
   case 8: samples_log2 = 3; break;
   default: REFUSE(EINVAL, "render sample count");
   }

   if ((uint64_t)c.sample_size_B * c.utile_width_px * c.utile_height_px *
             c.samples > M5_MAX_TILE_BUFFER ||
       !c.vdm_ctrl_stream_base || (c.vdm_ctrl_stream_base & 3) ||
       !c.isp_scissor_base || (c.isp_scissor_base & 7) ||
       (c.isp_dbias_base & 7) || (c.isp_oclqry_base & 7))
      REFUSE(EINVAL, "render tile buffer, VDM, scissor or depth-bias base");

   if (!zls_valid(&c.depth, c.layers) || !zls_valid(&c.stencil, c.layers))
      REFUSE(EINVAL, "render depth/stencil buffer");

   /* sampler(): the M5 command set has no sampler heap. */
   if (c.sampler_heap || c.sampler_count)
      REFUSE(EOPNOTSUPP, "render sampler heap (the M5 command set has none)");

   if (!helper_empty(&c.vertex_helper) || !helper_empty(&c.fragment_helper))
      REFUSE(EINVAL, "render helper program (M5 has none)");

   if (!c.bg.usc || !c.eot.usc || !c.partial_bg.usc || !c.partial_eot.usc)
      REFUSE(EINVAL, "render background/end-of-tile program missing");

   /* backend.rs prepare_command: the M5 runtime plans every render from its
    * program storage list. */
   if (!have_resources)
      REFUSE(EINVAL, "render without DRM_ASAHI_SET_RENDER_RESOURCES "
                     "(g17/runtime/backend.rs prepare_command)");

   return m5_render_admission(&c, samples_log2, features, reason);
}

static int
validate_compute(const uint8_t *bytes, size_t size, const char **reason)
{
   struct drm_asahi_cmd_compute c = {0};
   memcpy(&c, bytes, MIN2(size, sizeof(c)));

   if ((c.flags & ~M5_COMPUTE_FLAGS) ||
       (c.cdm_ctrl_stream_base & 3) || (c.cdm_ctrl_stream_end & 3) ||
       c.cdm_ctrl_stream_end <= c.cdm_ctrl_stream_base ||
       !helper_empty(&c.helper))
      REFUSE(EINVAL, "compute flags, stream extent or helper");
   if (c.sampler_heap || c.sampler_count)
      REFUSE(EOPNOTSUPP, "compute sampler heap (the M5 command set has none)");
   return 0;
}

int
asahi_m5_submit_gate(const void *cmdbuf, size_t size, uint64_t features,
                     struct asahi_m5_gate_stats *stats, const char **reason)
{
   const uint8_t *bytes = cmdbuf;
   unsigned counts[2] = {0, 0};
   bool render_resources = false;
   size_t at = 0;
   int ret;

   *reason = NULL;
   while (at < size) {
      struct drm_asahi_cmd_header h;
      if (size - at < sizeof(h))
         REFUSE(EINVAL, "truncated command header");
      memcpy(&h, bytes + at, sizeof(h));
      at += sizeof(h);
      if (h.size > size - at)
         REFUSE(EINVAL, "command runs past the command buffer");
      const uint8_t *payload = bytes + at;
      at += h.size;

      const bool software = h.cmd_type != DRM_ASAHI_CMD_RENDER &&
                            h.cmd_type != DRM_ASAHI_CMD_COMPUTE;
      if (software && (h.vdm_barrier != DRM_ASAHI_BARRIER_NONE ||
                       h.cdm_barrier != DRM_ASAHI_BARRIER_NONE))
         REFUSE(EINVAL, "software command with a barrier");

      switch (h.cmd_type) {
      case DRM_ASAHI_CMD_RENDER:
      case DRM_ASAHI_CMD_COMPUTE:
         if ((h.vdm_barrier != DRM_ASAHI_BARRIER_NONE &&
              h.vdm_barrier > counts[0]) ||
             (h.cdm_barrier != DRM_ASAHI_BARRIER_NONE &&
              h.cdm_barrier > counts[1]))
            REFUSE(EINVAL, "barrier past the commands before it");
         if (counts[0] + counts[1] >= MAX_COMMANDS)
            REFUSE(EINVAL, "too many commands");
         if (h.cmd_type == DRM_ASAHI_CMD_RENDER) {
            counts[0]++;
            stats->renders++;
            ret = validate_render(payload, h.size, render_resources, features,
                                  reason);
         } else {
            counts[1]++;
            stats->computes++;
            ret = validate_compute(payload, h.size, reason);
         }
         if (ret)
            return ret;
         break;
      case DRM_ASAHI_SET_VERTEX_ATTACHMENTS:
      case DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS:
      case DRM_ASAHI_SET_COMPUTE_ATTACHMENTS:
         if (h.size % sizeof(struct drm_asahi_attachment))
            REFUSE(EINVAL, "attachment list");
         break;
      case DRM_ASAHI_SET_RENDER_RESOURCES:
      case DRM_ASAHI_SET_RENDER_RESOURCES_V2:
         render_resources = h.size != 0;
         ret = parse_render_programs(
            payload, h.size, h.cmd_type == DRM_ASAHI_SET_RENDER_RESOURCES_V2,
            stats, reason);
         if (ret)
            return ret;
         break;
      case DRM_ASAHI_SET_COMPUTE_RESOURCES:
         ret = parse_compute_programs(payload, h.size, stats, reason);
         if (ret)
            return ret;
         break;
      default:
         REFUSE(EINVAL, "unknown command type");
      }
   }

   if (!counts[0] && !counts[1])
      REFUSE(EINVAL, "submission with no hardware command");
   return 0;
}
