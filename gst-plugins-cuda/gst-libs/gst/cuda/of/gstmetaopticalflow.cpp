/**************************** Includes and Macros *****************************/

#include <gst/cuda/nvcodec/gstcudacontext.h>
#include <gst/cuda/of/gstcudaofoutputvectorgridsize.h>
#include <gst/cuda/of/gstmetaopticalflow.h>

/*
 * Just some setup for the GStreamer debug logger.
 *
 * - J.O.
 */
GST_DEBUG_CATEGORY_STATIC(gst_meta_optical_flow_debug);
#define GST_CAT_DEFAULT gst_meta_optical_flow_debug

/************************** Type/Struct Definitions ***************************/

/*************************** Function Declarations ****************************/

/**
 * \brief Initialises an instance of the GstMetaOpticalFlow metadata type.
 *
 * \details This method is used as an override for the `init` method for the
 * GstMetaOpticalFlow metadata type. Specifically, it sets the pointer to the
 * cv::cuda::GpuMat instance to a nullptr.
 *
 * \param[in,out] meta A pointer to the GstMetaOpticalFlow instance.
 * \param[in] params A pointer to a structure containing a list of parameters
 * passed to the init function.
 * \param[in] buf A pointer to the buffer that the GstMetaOpticalFlow instance
 * is being initialised on.
 *
 * \returns TRUE under all circumstances.
 */
static gboolean
gst_meta_optical_flow_init(GstMeta *meta, gpointer params, GstBuffer *buf);

/**
 * \brief Cleans up an instance of the GstMetaOpticalFlow metadata type.
 *
 * \details This method is used as an override for the `free` method for the
 * GstMetaOpticalFlow metadata type. Specifically, it deletes the pointer to
 * the the cv::cuda::GpuMat instance and sets it to nullptr.
 *
 * \param[in,out] meta A pointer to the GstMetaOpticalFlow instance.
 * \param[in] buf A pointer to the buffer that the GstMetaOpticalFlow instance
 * is being freed from.
 */
static void gst_meta_optical_flow_free(GstMeta *meta, GstBuffer *buf);

/**
 * \brief Performs a transformation function on an instance of the
 * GstMetaOpticalFlow metadata type.
 *
 * \details This method is used as an override for the `transform` method for
 * the GstMetaOpticalFlow metadata type. Specifically, it deals exclusively
 * with the "copy" transformation type.
 *
 * \details For the "copy" transformation type, a new instance of the
 * GstMetaOpticalFlow metadata type is created on the new buffer (transbuf). A
 * new instance of the cv::cuda::GpuMat is also created using the instance held
 * by the current GstMetaOpticalFlow instance . The pointer to this newly
 * created cv::cuda::GpuMat instance is then assigned to the new
 * GstMetaOpticalFlow instance.
 *
 * \param[in,out] transbuf The buffer to perform the "copy" transformation
 * onto.
 * \param[in] meta A pointer to the GstMetaOpticalFlow instance.
 * \param[in] buf A pointer to the buffer that the GstMetaOpticalFlow instance
 * is being transformed on.
 * \param[in] type The type of transformation function to perform.
 * \param[in] data A pointer to a structure containing a list of parameters
 * passed to the transform function.
 */
static gboolean gst_meta_optical_flow_transform(
    GstBuffer *transbuf,
    GstMeta *meta,
    GstBuffer *buf,
    GQuark type,
    gpointer data);

/**************************** Function Definitions ****************************/

extern GType gst_meta_optical_flow_api_get_type()
{
    static GType type;
    static const gchar *tags[] = {NULL};

    if(g_once_init_enter(&type))
    {
        GType _type = gst_meta_api_type_register("GstMetaOpticalFlowAPI", tags);
        g_once_init_leave(&type, _type);
    }

    return type;
}

extern const GstMetaInfo *gst_meta_optical_flow_get_info()
{
    static const GstMetaInfo *meta_optical_flow_info = NULL;

    GST_DEBUG_CATEGORY_INIT(
        gst_meta_optical_flow_debug,
        "GstMetaOpticalFlow",
        0,
        "GStreamer CUDA Optical Flow Metadata");

    if(g_once_init_enter((GstMetaInfo **)&meta_optical_flow_info))
    {
        const GstMetaInfo *mi = gst_meta_register(
            GST_META_OPTICAL_FLOW_API_TYPE,
            "GstMetaOpticalFlow",
            sizeof(GstMetaOpticalFlow),
            gst_meta_optical_flow_init,
            gst_meta_optical_flow_free,
            gst_meta_optical_flow_transform);
        g_once_init_leave(
            (GstMetaInfo **)&meta_optical_flow_info, (GstMetaInfo *)mi);
    }

    return meta_optical_flow_info;
}

static gboolean
gst_meta_optical_flow_init(GstMeta *meta, gpointer params, GstBuffer *buf)
{
    GST_DEBUG(
        "GstMetaOpticalFlow instance stored at %p initialised on the buffer at "
        "%p",
        meta,
        buf);

    GstMetaOpticalFlow *optical_flow_meta = (GstMetaOpticalFlow *)(meta);
    optical_flow_meta->context = NULL;
    optical_flow_meta->optical_flow_vectors = nullptr;
    optical_flow_meta->optical_flow_vector_grid_size
        = OPTICAL_FLOW_OUTPUT_VECTOR_GRID_SIZE_1;

    return TRUE;
}

static void gst_meta_optical_flow_free(GstMeta *meta, GstBuffer *buf)
{
    GST_DEBUG(
        "GstMetaOpticalFlow instance stored at %p freed from the buffer at "
        "%p",
        meta,
        buf);

    GstMetaOpticalFlow *optical_flow_meta = (GstMetaOpticalFlow *)(meta);

    if(optical_flow_meta->optical_flow_vectors != nullptr)
    {
        if(optical_flow_meta->context != NULL
           && gst_cuda_context_push(optical_flow_meta->context))
        {
            delete optical_flow_meta->optical_flow_vectors;
            optical_flow_meta->optical_flow_vectors = nullptr;
            gst_cuda_context_pop(NULL);
        }
    }

    gst_clear_object(&optical_flow_meta->context);
}

static gboolean gst_meta_optical_flow_transform(
    GstBuffer *transbuf,
    GstMeta *meta,
    GstBuffer *buf,
    GQuark type,
    gpointer data)
{
    GstMetaOpticalFlow *old_optical_flow_meta = (GstMetaOpticalFlow *)(meta);
    GstMetaOpticalFlow *new_optical_flow_meta = (GstMetaOpticalFlow *)(meta);
    gboolean result = TRUE;

    GST_DEBUG(
        "GstMetaOpticalFlow instance stored at %p is being transformed from "
        "the buffer at %p to the buffer at %p",
        meta,
        buf,
        transbuf);

    if(GST_META_TRANSFORM_IS_COPY(type))
    {
        new_optical_flow_meta = GST_META_OPTICAL_FLOW_ADD(transbuf);

        new_optical_flow_meta->context
            = GST_CUDA_CONTEXT(gst_object_ref(old_optical_flow_meta->context));

        if(gst_cuda_context_push(new_optical_flow_meta->context))
        {
            new_optical_flow_meta->optical_flow_vectors = new cv::cuda::GpuMat(
                *(old_optical_flow_meta->optical_flow_vectors));
            gst_cuda_context_pop(NULL);
        }

        new_optical_flow_meta->optical_flow_vector_grid_size
            = old_optical_flow_meta->optical_flow_vector_grid_size;
    }
    else
    {
        result = FALSE;
    }

    return result;
}

/******************************************************************************/
