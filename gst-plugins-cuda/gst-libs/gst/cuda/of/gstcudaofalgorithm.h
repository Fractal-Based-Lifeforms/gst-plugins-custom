#ifndef _CUDA_OF_ALGORITHM_H_
#define _CUDA_OF_ALGORITHM_H_

#include <glib-object.h>
#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_CUDA_OF_ALGORITHM (gst_cuda_of_algorithm_get_type())

/**
 * /brief An enumeration containing the list of CUDA-based optical flow
 * algorithms as supported by OpenCV.
 */
typedef enum _GstCudaOfAlgorithm
{
    /**
     * \brief The OpenCV implementation of the optical flow algorithm developed
     * by Gunnar Farneback.
     */
    OPTICAL_FLOW_ALGORITHM_FARNEBACK,
    /**
     * \brief The OpenCV implementation of v1 of the optical flow algorithm
     * developed by NVIDIA as made available via their Optical Flow SDK.
     */
    OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0,
    /**
     * \brief The OpenCV implementation of v2 of the optical flow algorithm
     * developed by NVIDIA as made available via their Optical Flow SDK.
     */
    OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0
} GstCudaOfAlgorithm;

/**
 * \brief Type creation/retrieval function for the GstCudaOfAlgorithm enum type.
 *
 * \details This function creates and registers the GstCudaOfAlgorithm enum type
 * for the first invocation. The GType instance for the GstCudaOfAlgorithm enum
 * type is then returned.
 *
 * \details For subsequent invocations, the GType instance for the
 * GstCudaOfAlgorithm enum type is returned immediately.
 *
 * \returns A GType instance representing the type information for the
 * GstCudaOfAlgorithm enum type.
 */
extern __attribute__((visibility("default"))) GType
gst_cuda_of_algorithm_get_type();

G_END_DECLS

#endif
