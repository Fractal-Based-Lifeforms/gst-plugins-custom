#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <thread>

#include <glib-object.h>
#include <gst/cuda/featureextractor/cudafeaturesarray.h>
#include <gst/cuda/featureextractor/cudafeaturescell.h>
#include <gtest/gtest.h>

namespace
{

    TEST(CUDAFeaturesArrayTest, TestConstructor)
    {
        CUDAFeaturesArray *array = cuda_features_array_new(400u);
        ASSERT_NE(array, nullptr);
        g_object_unref(array);
    }

    TEST(CUDAFeaturesArrayTest, TestProperties)
    {
        CUDAFeaturesArray *array = cuda_features_array_new(400u);
        ASSERT_NE(array, nullptr);

        guint length = 0u;

        g_object_get(
            array,
            // clang-format off
            "features-array-length", &length,
            // clang-format on
            NULL);

        EXPECT_EQ(length,  400u);

        g_object_set(
            array,
            // clang-format off
            "features-array-length", 300u,
            // clang-format on
            NULL);

        g_object_get(
            array,
            // clang-format off
            "features-array-length", &length,
            // clang-format on
            NULL);

        // "features-array-length" is a read-only property, so this should not
        // change from its initial constructed value.
        EXPECT_EQ(length, 400u);

        g_object_unref(array);
    }

    TEST(CUDAFeaturesArrayTest, TestCellRetrieval)
    {
        CUDAFeaturesArray *array = cuda_features_array_new(400u);
        ASSERT_NE(array, nullptr);

        CUDAFeaturesCell *cell = cuda_features_array_at(array, 100u);
        EXPECT_NE(cell, nullptr);
        EXPECT_EQ(G_OBJECT(cell)->ref_count, 2u);

        gfloat spatial_magnitude = 0.0f;

        g_object_get(
            cell,
            // clang-format off
            "spatial-magnitude", &spatial_magnitude,
            // clang-format on
            NULL);

        EXPECT_FLOAT_EQ(spatial_magnitude, 0.0f);

        guint original_pixels = ((1280u * 720u) / (20u * 20u));
        guint original_count = original_pixels / 2u;
        gfloat original_spatial_magnitude = 256.0f * (gfloat)(original_count);

        g_object_set(
            cell,
            // clang-format off
            "spatial-magnitude", original_spatial_magnitude,
            // clang-format on
            NULL);
        g_object_unref(cell);

        cell = cuda_features_array_at(array, 100u);
        EXPECT_NE(cell, nullptr);
        EXPECT_EQ(G_OBJECT(cell)->ref_count, 2u);

        g_object_get(
            cell,
            // clang-format off
            "spatial-magnitude", &spatial_magnitude,
            // clang-format on
            NULL);

        EXPECT_FLOAT_EQ(spatial_magnitude, original_spatial_magnitude);

        g_object_unref(cell);
        g_object_unref(array);
    }
}
