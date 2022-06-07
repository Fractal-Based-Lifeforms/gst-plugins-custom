#ifndef __CUDA_FEATURES_ARRAY_H__
#define __CUDA_FEATURES_ARRAY_H__

#include <glib-object.h>

G_BEGIN_DECLS

#define CUDA_TYPE_FEATURES_ARRAY cuda_features_array_get_type()

extern __attribute__((visibility("default"))) G_DECLARE_FINAL_TYPE(
    CUDAFeaturesArray,
    cuda_features_array,
    CUDA,
    FEATURES_ARRAY,
    GObject);

typedef struct _CUDAFeaturesArray
{
    GObject parent;
} CUDAFeaturesArray;

typedef struct _CUDAFeaturesArrayPrivate
{
    GPtrArray *features_array;
} CUDAFeaturesArrayPrivate;

typedef struct _CUDAFeaturesCell CUDAFeaturesCell;

/**
 * \brief Retrieves the set of features at the given index of the features
 * array.
 *
 * \param[in] self A CudaFeaturesArray GObject instance to get the set of
 * features at the requested row & column.
 * \param[in] idx The index of the features array to retrieve the set of
 * features.

 * \returns A pointer representing a reference to a CUDAFeaturesCell
 * instance.
 *
 * \notes As this results in an increase in the reference count of the
 * CUDAFeaturesCell object, this reference must be deleted via `g_object_unref`
 * once the object reference is no longer required.
 */
extern __attribute__((visibility("default"))) CUDAFeaturesCell *
cuda_features_array_at(CUDAFeaturesArray *self, guint32 idx);

/**
 * \brief Constructor function for the CUDAFeaturesArray GObject.
 *
 * \param[in] length The intended length of the new CUDAFeaturesArray GObject
 * instance.
 *
 * \returns A new CUDAFeaturesArray GObject instance.
 */
extern __attribute__((visibility("default"))) CUDAFeaturesArray *
cuda_features_array_new(guint32 length);

G_END_DECLS

#endif
