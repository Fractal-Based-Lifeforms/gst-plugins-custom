#ifndef __GST_META_ALGORITHM_FEATURES_H__
#define __GST_META_ALGORITHM_FEATURES_H__

#include <glib-object.h>
#include <gmodule.h>
#include <gst/gst.h>

#include <gst/cuda/featureextractor/cudafeaturescell.h>
#include <gst/cuda/featureextractor/cudafeaturesmatrix.h>

G_BEGIN_DECLS

#define GST_META_ALGORITHM_FEATURES_API_TYPE \
    (gst_meta_algorithm_features_api_get_type())
#define GST_META_ALGORITHM_FEATURES_ADD(buf)           \
    ((GstMetaAlgorithmFeatures *)(gst_buffer_add_meta( \
        buf, gst_meta_algorithm_features_get_info(), NULL)))
#define GST_META_ALGORITHM_FEATURES_GET(buf)           \
    ((GstMetaAlgorithmFeatures *)(gst_buffer_get_meta( \
        buf, gst_meta_algorithm_features_api_get_type())))

/**
 * \brief The structure for the GstMetaAlgorithmFeatures metadata type.
 *
 * \details This structure contains the structure to the parent GstMeta
 * type, and a pointer to a FeaturesMatrix instance. The FeaturesMatrix
 * instance will contain the 6 features (Count, Pixels, X0ToX1Magnitude,
 * X1ToX0Magnitude, Y0ToY1Magnitude, Y1ToY0Magnitude) extracted for each grid
 * cell of the frame in a 20x20 (by default) matrix.
 */
typedef struct _GstMetaAlgorithmFeatures
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

    CUDAFeaturesMatrix *features;
} GstMetaAlgorithmFeatures;

/**
 * \brief Type creation/retrieval function for the GstMetaAlgorithmFeatures
 * metadata type.
 *
 * \details This function creates and registers the GstMetaAlgorithmFeatures
 * metadata type for the first invocation. The GType instance for the
 * GstMetaAlgorithmFeatures metadata type is then returned.
 *
 * \details For subsequent invocations, the GType instance for the
 * GstMetaAlgorithmFeatures metadata type is returned immediately.
 *
 * \returns A GType instance representing the type information for the
 * GstMetaAlgorithmFeatures metadata type.
 */
extern __attribute__((visibility("default"))) GType
gst_meta_algorithm_features_api_get_type(void);

/**
 * \brief GstMetaInfo creation/retrieval function for the
 * GstMetaAlgorithmFeatures metadata type.
 *
 * \details This function creates and registers the GstMetaInfo instance for
 * the GstMetaAlgorithmFeatures metadata type for the first invocation. The
 * GstMetaInfo instance for the GstMetaAlgorithmFeatures metadata type is then
 * returned.
 *
 * \details For subsequent invocations, the GstMetaInfo instance for the
 * GstMetaAlgorithmFeatures metadata type is then returned immediately.
 *
 * \returns A GstMetaInfo instance representing the registration information
 * for the GstMetaAlgorithmFeatures metadata type.
 */
extern __attribute__((visibility("default"))) const GstMetaInfo *
gst_meta_algorithm_features_get_info(void);

G_END_DECLS

#endif
