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
    constexpr gfloat original_x0_to_x1_magnitude
        = 256.0f * ((gfloat)(original_count) / 4.0f);
    constexpr gfloat original_x1_to_x0_magnitude
        = 256.0f * ((gfloat)(original_count) / 4.0f);
    constexpr gfloat original_y0_to_y1_magnitude
        = 256.0f * ((gfloat)(original_count) / 4.0f);
    constexpr gfloat original_y1_to_y0_magnitude
        = 256.0f * ((gfloat)(original_count) / 4.0f);

    TEST(CUDAFeaturesCellTest, TestConstructor)
    {
        CUDAFeaturesCell *cell
            = (CUDAFeaturesCell *)(g_object_new(CUDA_TYPE_FEATURES_CELL, NULL));
        ASSERT_TRUE(cell != NULL);
        g_object_unref(cell);
    }

    TEST(CUDAFeaturesCellTest, TestProperties)
    {

        CUDAFeaturesCell *cell
            = (CUDAFeaturesCell *)(g_object_new(CUDA_TYPE_FEATURES_CELL, NULL));
        ASSERT_TRUE(cell != NULL);

        guint count = 0u;
        guint pixels = 0u;
        gfloat x0_to_x1_magnitude = 0.0f;
        gfloat x1_to_x0_magnitude = 0.0f;
        gfloat y0_to_y1_magnitude = 0.0f;
        gfloat y1_to_y0_magnitude = 0.0f;

        g_object_get(
            cell,
            // clang-format off
            "count", &count,
            "pixels", &pixels,
            "x0-to-x1-magnitude", &x0_to_x1_magnitude,
            "x1-to-x0-magnitude", &x1_to_x0_magnitude,
            "y0-to-y1-magnitude", &y0_to_y1_magnitude,
            "y1-to-y0-magnitude", &y1_to_y0_magnitude,
            // clang-format on
            NULL);

        ASSERT_TRUE(count == 0u);
        ASSERT_TRUE(pixels == 0u);
        ASSERT_TRUE(x0_to_x1_magnitude == 0.0f);
        ASSERT_TRUE(x1_to_x0_magnitude == 0.0f);
        ASSERT_TRUE(y0_to_y1_magnitude == 0.0f);
        ASSERT_TRUE(y1_to_y0_magnitude == 0.0f);

        g_object_set(
            cell,
            // clang-format off
            "count", original_count,
            "pixels", original_pixels,
            "x0-to-x1-magnitude", original_x0_to_x1_magnitude,
            "x1-to-x0-magnitude", original_x1_to_x0_magnitude,
            "y0-to-y1-magnitude", original_y0_to_y1_magnitude,
            "y1-to-y0-magnitude", original_y1_to_y0_magnitude,
            // clang-format on
            NULL);

        g_object_get(
            cell,
            // clang-format off
            "count", &count,
            "pixels", &pixels,
            "x0-to-x1-magnitude", &x0_to_x1_magnitude,
            "x1-to-x0-magnitude", &x1_to_x0_magnitude,
            "y0-to-y1-magnitude", &y0_to_y1_magnitude,
            "y1-to-y0-magnitude", &y1_to_y0_magnitude,
            // clang-format on
            NULL);

        ASSERT_TRUE(count == original_count);
        ASSERT_TRUE(pixels == original_pixels);
        ASSERT_TRUE(x0_to_x1_magnitude == original_x0_to_x1_magnitude);
        ASSERT_TRUE(x1_to_x0_magnitude == original_x1_to_x0_magnitude);
        ASSERT_TRUE(y0_to_y1_magnitude == original_y0_to_y1_magnitude);
        ASSERT_TRUE(y1_to_y0_magnitude == original_y1_to_y0_magnitude);

        g_object_unref(cell);
    }

    TEST(CUDAFeaturesCellTest, TestConstructorWithProperties)
    {
        CUDAFeaturesCell *cell = (CUDAFeaturesCell *)(g_object_new(
            CUDA_TYPE_FEATURES_CELL,
            // clang-format off
            "count", original_count,
            "pixels", original_pixels,
            "x0-to-x1-magnitude", original_x0_to_x1_magnitude,
            "x1-to-x0-magnitude", original_x1_to_x0_magnitude,
            "y0-to-y1-magnitude", original_y0_to_y1_magnitude,
            "y1-to-y0-magnitude", original_y1_to_y0_magnitude,
            // clang-format on
            NULL));
        ASSERT_TRUE(cell != NULL);

        guint count = 0u;
        guint pixels = 0u;
        gfloat x0_to_x1_magnitude = 0.0f;
        gfloat x1_to_x0_magnitude = 0.0f;
        gfloat y0_to_y1_magnitude = 0.0f;
        gfloat y1_to_y0_magnitude = 0.0f;

        g_object_get(
            cell,
            // clang-format off
            "count", &count,
            "pixels", &pixels,
            "x0-to-x1-magnitude", &x0_to_x1_magnitude,
            "x1-to-x0-magnitude", &x1_to_x0_magnitude,
            "y0-to-y1-magnitude", &y0_to_y1_magnitude,
            "y1-to-y0-magnitude", &y1_to_y0_magnitude,
            // clang-format on
            NULL);

        ASSERT_TRUE(count == original_count);
        ASSERT_TRUE(pixels == original_pixels);
        ASSERT_TRUE(x0_to_x1_magnitude == original_x0_to_x1_magnitude);
        ASSERT_TRUE(x1_to_x0_magnitude == original_x1_to_x0_magnitude);
        ASSERT_TRUE(y0_to_y1_magnitude == original_y0_to_y1_magnitude);
        ASSERT_TRUE(y1_to_y0_magnitude == original_y1_to_y0_magnitude);

        g_object_unref(cell);
    }
}
