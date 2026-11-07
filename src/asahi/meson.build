# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# SPDX-License-Identifier: MIT

# This is used only for the r/e tooling on macOS
dep_iokit = dependency('IOKit', required : false)

inc_asahi = include_directories([
   '.', 'layout', 'lib', 'genxml', 'compiler'
])

subdir('isa')

with_asahi_common = with_gallium_asahi or with_asahi_vk or \
                    with_tools.contains('asahi')

if with_asahi_common
   subdir('genxml')
   subdir('libagx')
   subdir('compiler')
   subdir('clc')
endif

if with_gallium_asahi or with_asahi_vk
   subdir('layout')
   subdir('lib')
elif dep_iokit.found()
   # Just build enough for libwrap.dylib
   if not with_asahi_common
      subdir('genxml')
   endif
   subdir('lib')
endif

if with_asahi_vk
   subdir('vulkan')
endif

if with_tools.contains('drm-shim')
  subdir('drm-shim')

  if with_asahi_vk and with_tests and meson.can_run_host_binaries()
    dual_source_glslang = find_program('glslangValidator', required : false)
    if dual_source_glslang.found()
      test(
        'hk_dual_source_gpl',
        prog_python,
        args : [
          files('vulkan/tests/run-dual-source-gpl.py'),
          hk_dual_source_gpl_test, libvulkan_asahi, libasahi_noop_drm_shim,
          dual_source_glslang.full_path(),
        ],
        suite : ['asahi'],
      )
    endif
  endif
endif

if with_tools.contains('drm-shim') and with_asahi_vk and with_tests
  if prog_glslang.found()
    # Public extension query/pipeline/direct-record proof is compile-only under
    # the no-op shim: no command buffer is submitted to hardware here.
    test(
      'hk-g13-mesh-public-api',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_mesh_public_api_test,
        'supported',
      ],
      env : [
        'ASAHI_SHIM_CHIP=g13x',
        'ASAHI_G13_SOFTWARE_MESH=1',
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )

    # Record the complete public task/payload/mesh readback fixture against
    # the no-op G13X shim, but deliberately stop before queue submission.
    test(
      'hk-g13-mesh-task-readback-record',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_mesh_task_readback_test,
        '--record-only',
      ],
      env : [
        'ASAHI_SHIM_CHIP=g13x',
        'ASAHI_G13_SOFTWARE_MESH=1',
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
        'HK_TEST_ICD=' + libvulkan_asahi.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )

    # The stock G13 path is the environment-unset case, not merely an
    # explicit opt-out spelling.  Keep it separately executable so a default
    # change cannot accidentally turn the experimental extension on.
    test(
      'hk-g13-unset-mesh-public-api-negative',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_mesh_public_api_test,
        'unsupported',
      ],
      env : [
        'ASAHI_SHIM_CHIP=g13x',
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )

    foreach negative : [
      ['g13-disabled', 'g13x', '0'],
      ['g14', 'g14g', '1'],
    ]
      test(
        'hk-' + negative[0] + '-mesh-public-api-negative',
        prog_python,
        args : [
          files('../gallium/drivers/asahi/tests/run-shim.py'),
          libasahi_noop_drm_shim,
          hk_mesh_public_api_test,
          'unsupported',
        ],
        env : [
          'ASAHI_SHIM_CHIP=' + negative[1],
          'ASAHI_G13_SOFTWARE_MESH=' + negative[2],
          'VK_DRIVER_FILES=' + _dev_icd.full_path(),
        ],
        depends : [_dev_icd, libvulkan_asahi],
        suite : ['asahi'],
      )
    endforeach
  endif

  # Compile, upload and shader-binary-cache round-trip the G13 software-mesh
  # producer/pull-VS pair without advertising or executing mesh commands.
  test(
    'hk-g13-mesh-emulation-device-cache',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      hk_mesh_emulation_device_test,
    ],
    env : [
      'ASAHI_SHIM_CHIP=g13x',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Graphics-path copies from one layer of a block-compressed image into a
  # compressed G17P image; encoded and stream-validated, not executed.
  test(
    'hk-g17p-compressed-layer-copy',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_compressed_layer_copy,
      '--shim',
      '--expect-compressed',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_MESA_DEBUG=perf',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Mip chains on compressed images take compute dispatches, no render pass.
  test(
    'hk-g17p-compressed-mip-blit',
    prog_python,
    args : [
      files('../gallium/drivers/asahi/tests/run-shim.py'),
      libasahi_noop_drm_shim,
      g17p_vulkan_compressed_mip_blit,
      '--shim',
      '--expect-compute',
    ],
    env : [
      'ASAHI_SHIM_CHIP=g17p',
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'ASAHI_MESA_DEBUG=perf',
      'VK_DRIVER_FILES=' + _dev_icd.full_path(),
    ],
    depends : [_dev_icd, libvulkan_asahi],
    suite : ['asahi'],
  )

  # Explicit DRM plane layouts: placement, requirements, subresource layouts
  # and validation, without executing on the GPU.
  foreach chip : ['g13g', 'g17p']
    test(
      'hk-' + chip + '-explicit-plane-layout',
      prog_python,
      args : [
        files('../gallium/drivers/asahi/tests/run-shim.py'),
        libasahi_noop_drm_shim,
        hk_vulkan_explicit_plane_layout,
      ],
      env : [
        'ASAHI_SHIM_CHIP=' + chip,
        'VK_DRIVER_FILES=' + _dev_icd.full_path(),
      ],
      depends : [_dev_icd, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach
endif
