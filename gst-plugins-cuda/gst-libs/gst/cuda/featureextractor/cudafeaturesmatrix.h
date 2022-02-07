#ifndef __CUDA_FEATURES_MATRIX_H__
#define __CUDA_FEATURES_MATRIX_H__

#include <glib-object.h>

G_BEGIN_DECLS

#define CUDA_TYPE_FEATURES_MATRIX cuda_features_matrix_get_type()

extern __attribute__((visibility("default"))) G_DECLARE_FINAL_TYPE(
    CUDAFeaturesMatrix,
    cuda_features_matrix,
    CUDA,
    FEATURES_MATRIX,
    GObject);

typedef struct _CUDAFeaturesMatrix
{
    GObject parent;
} CUDAFeaturesMatrix;

typedef struct _CUDAFeaturesMatrixPrivate
{
    GPtrArray *features_matrix;
    guint32 features_matrix_cols;
    guint32 features_matrix_rows;
} CUDAFeaturesMatrixPrivate;

typedef struct _CUDAFeaturesCell CUDAFeaturesCell;

/**
 * \brief Retrieves the set of features at the given row & column of the
 * features matrix.
 *
 * \param[in] self A CudaFeaturesMatrix GObject instance to get the set of
 * features at the requested row & column.
 * \param[in] col The column of the features matrix to search for the set of
 * features.
 * \param[in] row The row of the features matrix to search for the set of
 * features.
 *
 * \returns A pointer representing a reference to a CUDAFeaturesMatrix
 * instance. This reference must be deleted via the `g_object_unref` function.
 *
 * \notes As this results in an increase in the reference count of the
 * CUDAFeaturesCell object, this reference must be deleted via `g_object_unref`
 * once the object reference is no longer required.
 */
extern __attribute__((visibility("default"))) CUDAFeaturesCell *
cuda_features_matrix_at(CUDAFeaturesMatrix *self, guint32 col, guint32 row);

G_END_DECLS

#endif
