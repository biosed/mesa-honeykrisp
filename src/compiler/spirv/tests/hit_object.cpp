/*
 * Copyright © 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "helpers.h"

class HitObject : public spirv_test {
protected:
   HitObject()
   {
      spirv_caps.RayQueryKHR = true;
      spirv_caps.RayTracingKHR = true;
      spirv_caps.ShaderInvocationReorderEXT = true;
   }
};

TEST_F(HitObject, record_from_query_operands)
{
   /* Generated with glslangValidator -V --target-env vulkan1.3 from:
    *
    * #version 460
    * #extension GL_EXT_ray_tracing : require
    * #extension GL_EXT_ray_query : require
    * #extension GL_EXT_shader_invocation_reorder : require
    * layout(location=0) hitObjectAttributeEXT vec2 attr0;
    * layout(location=1) hitObjectAttributeEXT vec4 attr1;
    * void main() {
    *  hitObjectEXT hit0, hit1;
    *  rayQueryEXT query;
    *  hitObjectRecordFromQueryEXT(hit0, query, 7, 0);
    *  hitObjectRecordFromQueryEXT(hit1, query, 9, 1, 0x42);
    *  hitObjectGetAttributesEXT(hit0, 0);
    *  hitObjectGetAttributesEXT(hit1, 1);
    * }
    */
   static const uint32_t words[] = {
      0x07230203, 0x00010600, 0x0008000b, 0x0000001b,
      0x00000000, 0x00020011, 0x00001178, 0x00020011,
      0x0000117f, 0x00020011, 0x0000150c, 0x000a000a,
      0x5f565053, 0x5f545845, 0x64616873, 0x695f7265,
      0x636f766e, 0x6f697461, 0x65725f6e, 0x6564726f,
      0x00000072, 0x0006000a, 0x5f565053, 0x5f52484b,
      0x5f796172, 0x72657571, 0x00000079, 0x0006000a,
      0x5f565053, 0x5f52484b, 0x5f796172, 0x63617274,
      0x00676e69, 0x0006000b, 0x00000001, 0x4c534c47,
      0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
      0x00000000, 0x00000001, 0x000a000f, 0x000014c1,
      0x00000004, 0x6e69616d, 0x00000000, 0x00000008,
      0x0000000b, 0x00000013, 0x00000014, 0x00000019,
      0x00030003, 0x00000002, 0x000001cc, 0x00060004,
      0x455f4c47, 0x725f5458, 0x715f7961, 0x79726575,
      0x00000000, 0x00060004, 0x455f4c47, 0x725f5458,
      0x745f7961, 0x69636172, 0x0000676e, 0x000a0004,
      0x455f4c47, 0x735f5458, 0x65646168, 0x6e695f72,
      0x61636f76, 0x6e6f6974, 0x6f65725f, 0x72656472,
      0x00000000, 0x00040005, 0x00000004, 0x6e69616d,
      0x00000000, 0x00040005, 0x00000008, 0x30746968,
      0x00000000, 0x00040005, 0x0000000b, 0x72657571,
      0x00000079, 0x00040005, 0x00000013, 0x72747461,
      0x00000030, 0x00040005, 0x00000014, 0x31746968,
      0x00000000, 0x00040005, 0x00000019, 0x72747461,
      0x00000031, 0x00020013, 0x00000002, 0x00030021,
      0x00000003, 0x00000002, 0x000214c1, 0x00000006,
      0x00040020, 0x00000007, 0x00000006, 0x00000006,
      0x0004003b, 0x00000007, 0x00000008, 0x00000006,
      0x00021178, 0x00000009, 0x00040020, 0x0000000a,
      0x00000006, 0x00000009, 0x0004003b, 0x0000000a,
      0x0000000b, 0x00000006, 0x00040015, 0x0000000c,
      0x00000020, 0x00000000, 0x0004002b, 0x0000000c,
      0x0000000d, 0x00000007, 0x00040015, 0x0000000e,
      0x00000020, 0x00000001, 0x0004002b, 0x0000000e,
      0x0000000f, 0x00000000, 0x00030016, 0x00000010,
      0x00000020, 0x00040017, 0x00000011, 0x00000010,
      0x00000002, 0x00040020, 0x00000012, 0x00001523,
      0x00000011, 0x0004003b, 0x00000012, 0x00000013,
      0x00001523, 0x0004003b, 0x00000007, 0x00000014,
      0x00000006, 0x0004002b, 0x0000000c, 0x00000015,
      0x00000009, 0x0004002b, 0x0000000e, 0x00000016,
      0x00000001, 0x00040017, 0x00000017, 0x00000010,
      0x00000004, 0x00040020, 0x00000018, 0x00001523,
      0x00000017, 0x0004003b, 0x00000018, 0x00000019,
      0x00001523, 0x0004002b, 0x0000000c, 0x0000001a,
      0x00000042, 0x00050036, 0x00000002, 0x00000004,
      0x00000000, 0x00000003, 0x000200f8, 0x00000005,
      0x000514b8, 0x00000008, 0x0000000b, 0x0000000d,
      0x00000013, 0x000614b8, 0x00000014, 0x0000000b,
      0x00000015, 0x00000019, 0x0000001a, 0x000314c9,
      0x00000008, 0x00000013, 0x000314c9, 0x00000014,
      0x00000019, 0x000100fd, 0x00010038,
   };

   get_nir(ARRAY_SIZE(words), words, MESA_SHADER_RAYGEN);

   ASSERT_NE(shader, nullptr);

   for (unsigned i = 0; i < 2; ++i) {
      nir_intrinsic_instr *record =
         find_intrinsic(nir_intrinsic_hit_object_record_from_query, i);
      ASSERT_NE(record, nullptr);
      ASSERT_TRUE(nir_src_is_const(record->src[2]));
      ASSERT_TRUE(nir_src_is_const(record->src[3]));
      EXPECT_EQ(nir_src_as_uint(record->src[2]), i ? 9u : 7u);
      EXPECT_EQ(nir_src_as_uint(record->src[3]), i ? 0x42u : 0u);
      EXPECT_EQ(record->num_components, i ? 4u : 2u);
      EXPECT_EQ(record->src[4].ssa->num_components, i ? 4u : 2u);

      /* The required attribute operand must load the selected location,
       * including when the optional hit kind is absent. */
      ASSERT_EQ(nir_def_instr(record->src[4].ssa)->type, nir_instr_type_intrinsic);
      nir_intrinsic_instr *attributes =
         nir_instr_as_intrinsic(nir_def_instr(record->src[4].ssa));
      ASSERT_EQ(attributes->intrinsic, nir_intrinsic_load_deref);
      nir_variable *var = nir_deref_instr_get_variable(
         nir_src_as_deref(attributes->src[0]));
      ASSERT_NE(var, nullptr);
      EXPECT_STREQ(var->name, i ? "attr1" : "attr0");

      nir_intrinsic_instr *get =
         find_intrinsic(nir_intrinsic_hit_object_load_attributes, i);
      ASSERT_NE(get, nullptr);
      EXPECT_EQ(get->def.num_components, i ? 4u : 2u);
      EXPECT_EQ(nir_deref_instr_get_variable(nir_src_as_deref(get->src[0])),
                nir_deref_instr_get_variable(nir_src_as_deref(record->src[0])));

      nir_intrinsic_instr *store = find_intrinsic(nir_intrinsic_store_deref, i);
      ASSERT_NE(store, nullptr);
      EXPECT_EQ(nir_deref_instr_get_variable(nir_src_as_deref(store->src[0])), var);
      EXPECT_EQ(store->src[1].ssa, &get->def);
   }
}
