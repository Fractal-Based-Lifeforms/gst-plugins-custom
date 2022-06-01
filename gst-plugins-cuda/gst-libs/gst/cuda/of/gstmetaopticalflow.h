#ifndef __GST_META_OPTICAL_FLOW_H__
#define __GST_META_OPTICAL_FLOW_H__

#include <glib-object.h>
#include <gmodule.h>
#include <gst/cuda/nvcodec/gstcudacontext.h>
#include <gst/gst.h>
#include <opencv2/core/cuda.hpp>

G_BEGIN_DECLS

#define GST_META_OPTICAL_FLOW_API_TYPE (gst_meta_optical_flow_api_get_type())
#define GST_META_OPTICAL_FLOW_ADD(buf)           \
    ((GstMetaOpticalFlow *)(gst_buffer_add_meta( \
        buf, gst_meta_optical_flow_get_info(), NULL)))
#define GST_META_OPTICAL_FLOW_GET(buf)           \
    ((GstMetaOpticalFlow *)(gst_buffer_get_meta( \
        buf, gst_meta_optical_flow_api_get_type())))

typedef struct _GstMetaOpticalFlow GstMetaOpticalFlow;

/**
 * \brief The structure for the GstMetaOpticalFlow metadata type.
 *
 * \details This structure contains the structure to the parent GstMeta type,
 * and a pointer to a cv::cuda::GpuMat instance. In particular, the
 * cv::cuda::GpuMat instance will contain a 2-channel 2D matrix of 32-bit
 * floating point values representing the output of the optical flow
 * algorithms.
 */
struct _GstMetaOpticalFlow
{
    /**
     * \brief The parent class' instance data.
     *
     * \details This is the structure for the parent class' instance data. When
     * a pointer to an instance of this class is cast to an instance to the
     * parent class or any other classes higher up in the class hierarchy, only
     * the variables available to that class will be available to be modified
     * or used.
     *
     * \notes As per above, this relies on a bit of trickery regarding how C
     * stores its data structures in memory. The order that the structures are
     * defined here are the order they will be stored in memory by C. That
     * allows us to "cheat" by casting a pointer to this structure to a pointer
     * of the parent structure(s); thereby giving us an inheritance-like
     * nature to these structures.
     */
    GstMeta meta;

    /**
     * \brief A pointer to the CUDA context that is intended to be pushed prior
     * to interacting with the 2D matrix held by the metadata.
     */
    GstCudaContext *context;

    /**
     * \brief A pointer to a 2-channel 2D matrix of optical flow values as
     * hosted on the GPU.
     */
    cv::cuda::GpuMat *optical_flow_vectors;

    /**
     * \brief An integer value representing the vector grid size of the optical
     * flow values.
     *
     * \notes This is required when dealing with some form of sparse optical
     * flow algorithm - primarily the NVIDIA Optical Flow algorithms - in order
     * to allow the feature-extractor to properly map optical-flow vectors to
     * pixels on the frame.
     */
    gint optical_flow_vector_grid_size;
};

/**
 * \brief Type creation/retrieval function for the GstMetaOpticalFlow metadata
 * type.
 *
 * \details This function creates and registers the GstMetaOpticalFlow metadata
 * type for the first invocation. The GType instance for the GstMetaOpticalFlow
 * metadata type is then returned.
 *
 * \details For subsequent invocations, the GType instance for the
 * GstMetaOpticalFlow metadata type is returned immediately.
 *
 * \returns A GType instance representing the type information for the
 * GstMetaOpticalFlow metadata type.
 */
extern __attribute__((visibility("default"))) GType
gst_meta_optical_flow_api_get_type();

/**
 * \brief GstMetaInfo creation/retrieval function for the GstMetaOpticalFlow
 * metadata type.
 *
 * \details This function creates and registers the GstMetaInfo instance for
 * the GstMetaOpticalFlow metadata type for the first invocation. The
 * GstMetaInfo instance for the GstMetaOpticalFlow metadata type is then
 * returned.
 *
 * \details For subsequent invocations, the GstMetaInfo instance for the
 * GstMetaOpticalFlow metadata type is then returned immediately.
 *
 * \returns A GstMetaInfo instance representing the registration information
 * for the GstMetaOpticalFlow metadata type.
 */
extern __attribute__((visibility("default"))) const GstMetaInfo *
gst_meta_optical_flow_get_info();

G_END_DECLS

#endif
