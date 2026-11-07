/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "compiler/nir/nir.h"
#include "util/mesa-blake3.h"

/*
 * Driver-internal parameter block for a software mesh producer.  The command
 * path will place its address in draw.geometry_params before dispatching the
 * lowered shader.  Keep this ABI independent of Vulkan application memory.
 */
struct hk_mesh_emulation_params {
   uint64_t vertex_buffer;
   uint64_t index_buffer;
};

struct hk_mesh_emulation_layout {
   unsigned shared_base;
   unsigned vertex_count;
   unsigned primitive_count;
   unsigned position;
   unsigned indices;
   unsigned shared_size;

   unsigned vertices_per_group;
   unsigned primitives_per_group;
};

/*
 * Immutable, cache-addressable result of admitting one API mesh shader to
 * the bounded G13 software contract.  It owns serialized NIR for both halves
 * of the pair so pipeline compilation can make private mutable clones and
 * still pass them through the ordinary Honeykrisp compiler.  In particular,
 * this object is not an uploaded shader and must never be submitted directly.
 */
struct hk_mesh_emulation_admission;

/*
 * Lower the bounded triangle-only mesh producer contract to a compute shader.
 *
 * The extension remains unadvertised on G13 until the command and generated
 * pull-vertex consumer paths are complete.  Returning false is fail-closed:
 * unsupported shaders are not modified.
 */
bool hk_nir_lower_mesh_emulation(
   nir_shader *shader, struct hk_mesh_emulation_layout *layout_out);

/*
 * Build the ordinary vertex-stage consumer for the position scratch.  The
 * matching U32 index scratch is consumed by the fixed-function indexed draw;
 * its fetched index arrives here as load_vertex_id.
 */
nir_shader *hk_nir_mesh_emulation_pull_vs(
   const nir_shader_compiler_options *options);

/*
 * Build an immutable producer/pull-VS pair.  The identity covers the contract
 * version and both serialized NIR programs.  Unsupported input returns NULL
 * without changing the caller's shader.
 */
struct hk_mesh_emulation_admission *
hk_mesh_emulation_admission_create(
   const nir_shader *mesh, const nir_shader_compiler_options *options);

struct hk_mesh_emulation_admission *
hk_mesh_emulation_admission_ref(struct hk_mesh_emulation_admission *admission);

/* Returns true when this call released the final reference. */
bool hk_mesh_emulation_admission_unref(
   struct hk_mesh_emulation_admission *admission);

const uint8_t *hk_mesh_emulation_admission_identity(
   const struct hk_mesh_emulation_admission *admission);

const struct hk_mesh_emulation_layout *hk_mesh_emulation_admission_layout(
   const struct hk_mesh_emulation_admission *admission);

nir_shader *hk_mesh_emulation_admission_clone_producer(
   const struct hk_mesh_emulation_admission *admission,
   const nir_shader_compiler_options *options);

nir_shader *hk_mesh_emulation_admission_clone_pull_vs(
   const struct hk_mesh_emulation_admission *admission,
   const nir_shader_compiler_options *options);
