#ifndef _CUDA_OF_H_
#define _CUDA_OF_H_

#include <glib-object.h>
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

G_BEGIN_DECLS

// clang-format off
/**
 * /brief An enumeration containing the list of CUDA-based optical flow
 * algorithms as supported by OpenCV.
 */
// clang-format on
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

// clang-format off
/**
 * \brief Type creation/retrieval function for the GstCudaOf object type.
 *
 * \details This function creates and registers the GstCudaOf object type for
 * the first invocation. The GType instance for the GstCudaOf object type is
 * then returned.
 *
 * \details For subsequent invocations, the GType instance for the GstCudaOf
 * object type is returned immediately.
 *
 * \returns A GType instance representing the type information for the
 * GstCudaOf object type.
 */
// clang-format on
GType gst_cuda_of_get_type();

// clang-format off
/**
 * \brief Special initialisation function for GStreamer plugins.
 *
 * \details This is an initialisation function that is called when this plugin
 * library is loaded by GStreamer. This sets up the necessary element
 * registrations and logging categories for the plugin.
 *
 * \param[in,out] plugin The loaded GStreamer plugin to register the plugin's
 * elements with.
 */
// clang-format on
gboolean gst_cuda_of_plugin_init(GstPlugin *plugin);

G_END_DECLS

#endif
