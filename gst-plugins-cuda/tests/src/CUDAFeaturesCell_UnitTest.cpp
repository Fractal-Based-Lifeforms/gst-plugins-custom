#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <thread>

#include <glib-object.h>
#include <gst/cuda/featureextractor/cudafeaturescell.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using ::testing::Combine;
using ::testing::Values;

namespace
{

    constexpr guint original_pixels = ((1280u * 720u) / (20u * 20u));
    constexpr guint original_count = original_pixels / 2u;
    constexpr gfloat original_spatial_magnitude
        = 256.0f * (gfloat)(original_count);

    TEST(CUDAFeaturesCellTest, TestConstructor)
    {
        CUDAFeaturesCell *cell
            = (CUDAFeaturesCell *)(g_object_new(CUDA_TYPE_FEATURES_CELL, NULL));
        ASSERT_NE(cell, nullptr);
        g_object_unref(cell);
    }

    TEST(CUDAFeaturesCellTest, TestProperties)
    {

        CUDAFeaturesCell *cell
            = (CUDAFeaturesCell *)(g_object_new(CUDA_TYPE_FEATURES_CELL, NULL));
        ASSERT_NE(cell, nullptr);

        gfloat spatial_magnitude = 0.0f;

        g_object_get(
            cell,
            // clang-format off
            "spatial-magnitude", &spatial_magnitude,
            // clang-format on
            NULL);

        EXPECT_TRUE(spatial_magnitude == 0.0f);

        g_object_set(
            cell,
            // clang-format off
            "spatial-magnitude", original_spatial_magnitude,
            // clang-format on
            NULL);

        g_object_get(
            cell,
            // clang-format off
            "spatial-magnitude", &spatial_magnitude,
            // clang-format on
            NULL);

        EXPECT_FLOAT_EQ(spatial_magnitude, original_spatial_magnitude);

        g_object_unref(cell);
    }

    TEST(CUDAFeaturesCellTest, TestConstructorWithProperties)
    {
        CUDAFeaturesCell *cell = (CUDAFeaturesCell *)(g_object_new(
            CUDA_TYPE_FEATURES_CELL,
            // clang-format off
            "spatial-magnitude", original_spatial_magnitude,
            // clang-format on
            NULL));
        ASSERT_NE(cell, nullptr);

        gfloat spatial_magnitude = 0.0f;

        g_object_get(
            cell,
            // clang-format off
            "spatial-magnitude", &spatial_magnitude,
            // clang-format on
            NULL);

        EXPECT_FLOAT_EQ(spatial_magnitude, original_spatial_magnitude);

        g_object_unref(cell);
    }
}
