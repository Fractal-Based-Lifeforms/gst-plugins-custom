#ifndef __CUDA_FEATURES_CELL_H__
#define __CUDA_FEATURES_CELL_H__

#include <glib-object.h>

G_BEGIN_DECLS

#define CUDA_TYPE_FEATURES_CELL cuda_features_cell_get_type()

extern __attribute__((visibility("default"))) G_DECLARE_FINAL_TYPE(
    CUDAFeaturesCell,
    cuda_features_cell,
    CUDA,
    FEATURES_CELL,
    GObject);

typedef struct _CUDAFeaturesCell
{
    GObject parent;
} CUDAFeaturesCell;

typedef struct _CUDAFeaturesCellPrivate
{
    gfloat spatial_magnitude;
} CUDAFeaturesCellPrivate;

G_END_DECLS

#endif
