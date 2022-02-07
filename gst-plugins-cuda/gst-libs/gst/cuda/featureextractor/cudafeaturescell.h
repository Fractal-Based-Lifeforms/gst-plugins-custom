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
    guint32 count;
    guint32 pixels;
    gfloat x0_to_x1_magnitude;
    gfloat x1_to_x0_magnitude;
    gfloat y0_to_y1_magnitude;
    gfloat y1_to_y0_magnitude;
} CUDAFeaturesCellPrivate;

G_END_DECLS

#endif
