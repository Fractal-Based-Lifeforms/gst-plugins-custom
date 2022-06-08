/**************************** Includes and Macros *****************************/

#include <gst/cuda/featureextractor/cudafeaturesarray.h>
#include <gst/cuda/featureextractor/gstmetaalgorithmfeatures.h>

/*
 * Just some setup for the GStreamer debug logger.
 *
 * - J.O.
 */
GST_DEBUG_CATEGORY_STATIC(gst_meta_algorithm_features_debug);
#define GST_CAT_DEFAULT gst_meta_algorithm_features_debug

/************************** Type/Struct Definitions ***************************/

/*************************** Function Declarations ****************************/

/**
 * \brief Initialises an instance of the GstMetaAlgorithmFeatures metadata type.
 *
 * \details This method is used as an override for the `init` method for the
 * GstMetaAlgorithmFeatures metadata type. Specifically, it sets the pointer to
 * the FeatureExtractorCUDA::FeaturesMatrix instance to a nullptr.
 *
 * \param[in,out] meta A pointer to the GstMetaAlgorithmFeatures instance.
 * \param[in] params A pointer to a structure containing a list of parameters
 * passed to the init function.
 * \param[in] buf A pointer to the buffer that the GstMetaAlgorithmFeatures
 * instance is being initialised on.
 *
 * \returns TRUE under all circumstances.
 */
static gboolean gst_meta_algorithm_features_init(
    GstMeta *meta,
    gpointer params,
    GstBuffer *buf);

/**
 * \brief Cleans up an instance of the GstMetaAlgorithmFeatures metadata type.
 *
 * \details This method is used as an override for the `free` method for the
 * GstMetaAlgorithmFeatures metadata type. Specifically, it deletes the pointer
 * to the the FeatureExtractorCUDA::FeaturesMatrix instance and sets it to
 * nullptr.
 *
 * \param[in,out] meta A pointer to the GstMetaAlgorithmFeatures instance.
 * \param[in] buf A pointer to the buffer that the GstMetaAlgorithmFeatures
 * instance is being freed from.
 */
static void gst_meta_algorithm_features_free(GstMeta *meta, GstBuffer *buf);

/**
 * \brief Performs a transformation function on an instance of the
 * GstMetaAlgorithmFeatures metadata type.
 *
 * \details This method is used as an override for the `transform` method for
 * the GstMetaAlgorithmFeatures metadata type. Specifically, it deals
 * exclusively with the "copy" transformation type.
 *
 * \details For the "copy" transformation type, a new instance of the
 * GstMetaAlgorithmFeatures metadata type is created on the new buffer
 * (transbuf). The pointer to the existing FeatureExtractorCUDA::FeaturesMatrix
 * instance is then assigned to the new GstMetaAlgorithmFeatures instance,
 * increasing the reference count to the FeatureExtractorCUDA::FeaturesMatrix
 * instance.
 *
 * \param[in,out] transbuf The buffer to perform the "copy" transformation
 * onto.
 * \param[in] meta A pointer to the GstMetaAlgorithmFeatures instance.
 * \param[in] buf A pointer to the buffer that the GstMetaAlgorithmFeatures
 * instance is being transformed on.
 * \param[in] type The type of transformation function to perform.
 * \param[in] data A pointer to a structure containing a list of parameters
 * passed to the transform function.
 */
static gboolean gst_meta_algorithm_features_transform(
    GstBuffer *transbuf,
    GstMeta *meta,
    GstBuffer *buf,
    GQuark type,
    gpointer data);

/****************************** Static Variables ******************************/

/************************** GObject Type Definitions **************************/

/**************************** Function Definitions ****************************/

GType gst_meta_algorithm_features_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = {NULL};

    if(g_once_init_enter(&type))
    {
        GType _type
            = gst_meta_api_type_register("GstMetaAlgorithmFeaturesAPI", tags);
        g_once_init_leave(&type, _type);
    }

    return type;
}

const GstMetaInfo *gst_meta_algorithm_features_get_info(void)
{
    static const GstMetaInfo *meta_algorithm_features_info = NULL;

    GST_DEBUG_CATEGORY_INIT(
        gst_meta_algorithm_features_debug,
        "GstMetaAlgorithmFeatures",
        0,
        "GStreamer CUDA Algorithm Features Metadata");

    if(g_once_init_enter((GstMetaInfo **)&meta_algorithm_features_info))
    {
        const GstMetaInfo *mi = gst_meta_register(
            GST_META_ALGORITHM_FEATURES_API_TYPE,
            "GstMetaAlgorithmFeatures",
            sizeof(GstMetaAlgorithmFeatures),
            gst_meta_algorithm_features_init,
            gst_meta_algorithm_features_free,
            gst_meta_algorithm_features_transform);
        g_once_init_leave(
            (GstMetaInfo **)&meta_algorithm_features_info, (GstMetaInfo *)mi);
    }

    return meta_algorithm_features_info;
}

static gboolean
gst_meta_algorithm_features_init(GstMeta *meta, gpointer params, GstBuffer *buf)
{
    GstMetaAlgorithmFeatures *algorithm_features_meta
        = (GstMetaAlgorithmFeatures *)(meta);

    GST_DEBUG(
        "GstMetaAlgorithmFeatures instance stored at %p initialised on the "
        "buffer at "
        "%p",
        meta,
        buf);

    algorithm_features_meta->features = NULL;

    return TRUE;
}

static void gst_meta_algorithm_features_free(GstMeta *meta, GstBuffer *buf)
{
    GstMetaAlgorithmFeatures *algorithm_features_meta
        = (GstMetaAlgorithmFeatures *)(meta);

    GST_DEBUG(
        "GstMetaAlgorithmFeatures instance stored at %p freed from the buffer "
        "at "
        "%p",
        meta,
        buf);

    if(algorithm_features_meta->features != NULL)
    {
        g_array_unref(algorithm_features_meta->features);
        algorithm_features_meta->features = NULL;
    }
}

static gboolean gst_meta_algorithm_features_transform(
    GstBuffer *transbuf,
    GstMeta *meta,
    GstBuffer *buf,
    GQuark type,
    gpointer data)
{
    GstMetaAlgorithmFeatures *old_algorithm_features_meta
        = (GstMetaAlgorithmFeatures *)(meta);
    GstMetaAlgorithmFeatures *new_algorithm_features_meta
        = (GstMetaAlgorithmFeatures *)(meta);
    gboolean result = TRUE;

    GST_DEBUG(
        "GstMetaAlgorithmFeatures instance stored at %p is being transformed "
        "from "
        "the buffer at %p to the buffer at %p",
        meta,
        buf,
        transbuf);

    if(GST_META_TRANSFORM_IS_COPY(type))
    {
        new_algorithm_features_meta = GST_META_ALGORITHM_FEATURES_ADD(transbuf);

        if(old_algorithm_features_meta->features != NULL)
        {
            new_algorithm_features_meta->features
                = g_array_ref(old_algorithm_features_meta->features);
        }
    }
    else
    {
        result = FALSE;
    }

    return result;
}

/******************************************************************************/
