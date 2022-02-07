#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <thread>

#include <glib-object.h>
#include <gst/cuda/featureextractor/cudafeaturescell.h>
#include <gst/cuda/featureextractor/cudafeaturesmatrix.h>
#include <gtest/gtest.h>

namespace
{

    TEST(CUDAFeaturesMatrixTest, TestConstructor)
    {
        CUDAFeaturesMatrix *matrix = (CUDAFeaturesMatrix *)(g_object_new(
            CUDA_TYPE_FEATURES_MATRIX, NULL));
        ASSERT_TRUE(matrix != NULL);
        g_object_unref(matrix);
    }

    TEST(CUDAFeaturesMatrixTest, TestProperties)
    {
        CUDAFeaturesMatrix *matrix = (CUDAFeaturesMatrix *)(g_object_new(
            CUDA_TYPE_FEATURES_MATRIX, NULL));
        ASSERT_TRUE(matrix != NULL);

        guint rows = 0u;
        guint cols = 0u;

        g_object_get(
            matrix,
            // clang-format off
            "features-matrix-rows", &rows,
            "features-matrix-cols", &cols,
            // clang-format on
            NULL);

        ASSERT_TRUE(rows == 20u);
        ASSERT_TRUE(cols == 20u);

        g_object_set(
            matrix,
            // clang-format off
            "features-matrix-rows", 30u,
            "features-matrix-cols", 30u,
            // clang-format on
            NULL);

        g_object_get(
            matrix,
            // clang-format off
            "features-matrix-rows", &rows,
            "features-matrix-cols", &cols,
            // clang-format on
            NULL);

        ASSERT_TRUE(rows == 20u);
        ASSERT_TRUE(cols == 20u);

        g_object_unref(matrix);
    }

    TEST(CUDAFeaturesMatrixTest, TestConstructorWithProperties)
    {
        guint rows = 0u;
        guint cols = 0u;

        CUDAFeaturesMatrix *matrix = (CUDAFeaturesMatrix *)(g_object_new(
            CUDA_TYPE_FEATURES_MATRIX,
            // clang-format off
            "features-matrix-rows", 30u,
            "features-matrix-cols", 30u,
            // clang-format on
            NULL));
        ASSERT_TRUE(matrix != NULL);

        g_object_get(
            matrix,
            // clang-format off
            "features-matrix-rows", &rows,
            "features-matrix-cols", &cols,
            // clang-format on
            NULL);

        ASSERT_TRUE(rows == 30u);
        ASSERT_TRUE(cols == 30u);

        g_object_unref(matrix);
    }

    TEST(CUDAFeaturesMatrixTest, TestCellRetrieval)
    {
        CUDAFeaturesMatrix *matrix = (CUDAFeaturesMatrix *)(g_object_new(
            CUDA_TYPE_FEATURES_MATRIX, NULL));
        ASSERT_TRUE(matrix != NULL);

        CUDAFeaturesCell *cell = cuda_features_matrix_at(matrix, 10, 10);
        ASSERT_TRUE(cell != NULL);
        ASSERT_TRUE(G_OBJECT(cell)->ref_count == 2);

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

        guint original_pixels = ((1280u * 720u) / (20u * 20u));
        guint original_count = original_pixels / 2u;
        gfloat original_x0_to_x1_magnitude
            = 256.0f * ((gfloat)(original_count) / 4.0f);
        gfloat original_x1_to_x0_magnitude
            = 256.0f * ((gfloat)(original_count) / 4.0f);
        gfloat original_y0_to_y1_magnitude
            = 256.0f * ((gfloat)(original_count) / 4.0f);
        gfloat original_y1_to_y0_magnitude
            = 256.0f * ((gfloat)(original_count) / 4.0f);

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
        g_object_unref(cell);

        cell = cuda_features_matrix_at(matrix, 10u, 10u);
        ASSERT_TRUE(cell != NULL);
        ASSERT_TRUE(G_OBJECT(cell)->ref_count == 2u);

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
        g_object_unref(matrix);
    }
}
