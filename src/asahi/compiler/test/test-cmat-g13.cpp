/* Copyright © 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>

#include "agx_test.h"

extern "C" {
#include "agx_cmat_layout.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
}

TEST(G13CooperativeMatrix, LayoutIsBijective)
{
   bool seen[8][8] = {};

   for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned element = 0; element < 2; ++element) {
         unsigned row = agx_cmat_8x8_row(lane, element);
         unsigned col = agx_cmat_8x8_col(lane, element);

         ASSERT_LT(row, 8u);
         ASSERT_LT(col, 8u);
         EXPECT_FALSE(seen[row][col]);
         seen[row][col] = true;
         EXPECT_EQ(agx_cmat_8x8_owner_lane(row, col), lane);
         EXPECT_EQ(agx_cmat_8x8_owner_element(row, col), element);
      }
   }

   for (unsigned row = 0; row < 8; ++row)
      for (unsigned col = 0; col < 8; ++col)
         EXPECT_TRUE(seen[row][col]);
}

TEST(G13CooperativeMatrix, DistributedTileMatchesReference)
{
   float a[8][8], b[8][8], c[8][8], expected[8][8];
   float lane_a[32][2], lane_b[32][2], lane_c[32][2];

   for (unsigned row = 0; row < 8; ++row) {
      for (unsigned col = 0; col < 8; ++col) {
         /* Small integers are exact in both FP16 and FP32. */
         a[row][col] = (int)((row * 3 + col * 5) % 7) - 3;
         b[row][col] = (int)((row * 5 + col * 2) % 9) - 4;
         c[row][col] = (int)((row + col * 3) % 5) - 2;
      }
   }

   for (unsigned row = 0; row < 8; ++row) {
      for (unsigned col = 0; col < 8; ++col) {
         expected[row][col] = c[row][col];
         for (unsigned k = 0; k < 8; ++k)
            expected[row][col] += a[row][k] * b[k][col];
      }
   }

   for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned element = 0; element < 2; ++element) {
         unsigned row = agx_cmat_8x8_row(lane, element);
         unsigned col = agx_cmat_8x8_col(lane, element);
         lane_a[lane][element] = a[row][col];
         lane_b[lane][element] = b[row][col];
         lane_c[lane][element] = c[row][col];
      }
   }

   for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned element = 0; element < 2; ++element) {
         unsigned row = agx_cmat_8x8_row(lane, element);
         unsigned col = agx_cmat_8x8_col(lane, element);
         float value = lane_c[lane][element];

         for (unsigned k = 0; k < 8; ++k) {
            unsigned a_lane = agx_cmat_8x8_owner_lane(row, k);
            unsigned a_element = agx_cmat_8x8_owner_element(row, k);
            unsigned b_lane = agx_cmat_8x8_owner_lane(k, col);
            unsigned b_element = agx_cmat_8x8_owner_element(k, col);
            value += lane_a[a_lane][a_element] *
                     lane_b[b_lane][b_element];
         }

         EXPECT_FLOAT_EQ(value, expected[row][col]);
      }
   }
}

class G13CooperativeMatrixNIR : public ::testing::Test {
protected:
   void SetUp() override { glsl_type_singleton_init_or_ref(); }
   void TearDown() override { glsl_type_singleton_decref(); }
};

static nir_variable *
cmat_var(nir_builder *b, enum glsl_base_type element_type,
         enum glsl_cmat_use use, const char *name)
{
   struct glsl_cmat_description desc = {};
   desc.rows = 8;
   desc.cols = 8;
   desc.element_type = element_type;
   desc.scope = SCOPE_SUBGROUP;
   desc.use = use;
   return nir_local_variable_create(b->impl, glsl_cmat_type(&desc), name);
}

static unsigned
count_intrinsic(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            count += instr->type == nir_instr_type_intrinsic &&
                     nir_instr_as_intrinsic(instr)->intrinsic == op;
         }
      }
   }
   return count;
}

static unsigned
count_alu(nir_shader *shader, nir_op op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            count += instr->type == nir_instr_type_alu &&
                     nir_instr_as_alu(instr)->op == op;
         }
      }
   }
   return count;
}

TEST_F(G13CooperativeMatrixNIR, MuladdLowersToShufflesAndFma)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "g13 cooperative matrix tile");
   b.shader->info.cs.has_cooperative_matrix = true;
   b.shader->info.workgroup_size[0] = 32;

   nir_variable *a = cmat_var(&b, GLSL_TYPE_FLOAT16, GLSL_CMAT_USE_A, "a");
   nir_variable *bb = cmat_var(&b, GLSL_TYPE_FLOAT16, GLSL_CMAT_USE_B, "b");
   nir_variable *c = cmat_var(&b, GLSL_TYPE_FLOAT,
                              GLSL_CMAT_USE_ACCUMULATOR, "c");
   nir_variable *result = cmat_var(&b, GLSL_TYPE_FLOAT,
                                   GLSL_CMAT_USE_ACCUMULATOR, "result");
   nir_deref_instr *a_deref = nir_build_deref_var(&b, a);
   nir_deref_instr *b_deref = nir_build_deref_var(&b, bb);
   nir_deref_instr *c_deref = nir_build_deref_var(&b, c);
   nir_deref_instr *result_deref = nir_build_deref_var(&b, result);

   nir_cmat_construct(&b, &a_deref->def, nir_imm_floatN_t(&b, 1.0, 16));
   nir_cmat_construct(&b, &b_deref->def, nir_imm_floatN_t(&b, 2.0, 16));
   nir_cmat_construct(&b, &c_deref->def, nir_imm_float(&b, 3.0));
   nir_cmat_muladd(&b, &result_deref->def, &a_deref->def, &b_deref->def,
                   &c_deref->def);

   ASSERT_TRUE(agx_nir_lower_cmat_g13(b.shader));
   EXPECT_EQ(count_intrinsic(b.shader, nir_intrinsic_cmat_muladd), 0u);
   EXPECT_EQ(count_intrinsic(b.shader, nir_intrinsic_cmat_muladd_agx), 0u);
   EXPECT_EQ(count_intrinsic(b.shader, nir_intrinsic_shuffle), 32u);
   EXPECT_EQ(count_alu(b.shader, nir_op_ffma), 16u);
   ASSERT_TRUE(agx_nir_lower_subgroups(b.shader));
   EXPECT_EQ(count_intrinsic(b.shader, nir_intrinsic_read_invocation), 128u);
   nir_opt_dce(b.shader);
   EXPECT_EQ(count_intrinsic(b.shader, nir_intrinsic_shuffle), 0u);
   nir_validate_shader(b.shader, "G13 software cooperative matrix result");
   ralloc_free(b.shader);
}

TEST_F(G13CooperativeMatrixNIR, UnsupportedShapeFailsClosed)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unsupported cooperative matrix");
   b.shader->info.cs.has_cooperative_matrix = true;
   struct glsl_cmat_description desc = {};
   desc.rows = 16;
   desc.cols = 16;
   desc.element_type = GLSL_TYPE_FLOAT16;
   desc.scope = SCOPE_SUBGROUP;
   desc.use = GLSL_CMAT_USE_A;
   nir_local_variable_create(b.impl, glsl_cmat_type(&desc), "unsupported");

   EXPECT_FALSE(agx_nir_lower_cmat_g13(b.shader));
   nir_validate_shader(b.shader, "unsupported G13 cooperative matrix");
   ralloc_free(b.shader);
}

TEST_F(G13CooperativeMatrixNIR, SaturatedConvertFailsClosed)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "saturated matrix conversion");
   b.shader->info.cs.has_cooperative_matrix = true;
   nir_variable *src = cmat_var(&b, GLSL_TYPE_FLOAT,
                                GLSL_CMAT_USE_ACCUMULATOR, "src");
   nir_variable *dst = cmat_var(&b, GLSL_TYPE_FLOAT,
                                GLSL_CMAT_USE_ACCUMULATOR, "dst");
   nir_deref_instr *src_deref = nir_build_deref_var(&b, src);
   nir_deref_instr *dst_deref = nir_build_deref_var(&b, dst);
   nir_cmat_convert(&b, &dst_deref->def, &src_deref->def, .saturate = true);

   ASSERT_EQ(count_intrinsic(b.shader, nir_intrinsic_cmat_convert), 1u);
   EXPECT_FALSE(agx_nir_lower_cmat_g13(b.shader));
   EXPECT_EQ(count_intrinsic(b.shader, nir_intrinsic_cmat_convert), 1u);
   EXPECT_TRUE(glsl_type_is_cmat(src_deref->type));
   EXPECT_TRUE(glsl_type_is_cmat(dst_deref->type));
   nir_validate_shader(b.shader, "rejected saturated matrix conversion");
   ralloc_free(b.shader);
}

TEST_F(G13CooperativeMatrixNIR, UnhandledIntrinsicFailsClosed)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unhandled matrix operation");
   b.shader->info.cs.has_cooperative_matrix = true;
   nir_variable *src = cmat_var(&b, GLSL_TYPE_FLOAT,
                                GLSL_CMAT_USE_ACCUMULATOR, "src");
   nir_variable *dst = cmat_var(&b, GLSL_TYPE_FLOAT,
                                GLSL_CMAT_USE_ACCUMULATOR, "dst");
   nir_deref_instr *src_deref = nir_build_deref_var(&b, src);
   nir_deref_instr *dst_deref = nir_build_deref_var(&b, dst);
   nir_cmat_transpose(&b, &dst_deref->def, &src_deref->def);

   ASSERT_EQ(count_intrinsic(b.shader, nir_intrinsic_cmat_transpose), 1u);
   EXPECT_FALSE(agx_nir_lower_cmat_g13(b.shader));
   EXPECT_EQ(count_intrinsic(b.shader, nir_intrinsic_cmat_transpose), 1u);
   EXPECT_TRUE(glsl_type_is_cmat(src_deref->type));
   EXPECT_TRUE(glsl_type_is_cmat(dst_deref->type));
   nir_validate_shader(b.shader, "rejected unhandled matrix operation");
   ralloc_free(b.shader);
}
