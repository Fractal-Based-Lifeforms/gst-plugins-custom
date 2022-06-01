/**************************** Includes and Macros *****************************/

/*
 * Just an explanation for the below macro section:
 *
 * Up until GCC version 8, the support for the C++17 filesystem library within
 * the STL was considered experimental. Thus, we need a different include path
 * when dealing with older GCC versions.
 *
 * In addition, until GCC version 9, the C++17 filesystem library is separate
 * from the libstdc++ library and thus required linker flags to link the
 * library to the rest of the code.
 *
 * - J.O.
 */
#include "gstcudafeatureextractor.h"

#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(8, 0)
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#else
#include <filesystem>
#endif

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <glib-object.h>
#include <glibconfig.h>
#include <gst/base/gstbasetransform.h>
#include <gst/cuda/featureextractor/cudafeaturesmatrix.h>
#include <gst/cuda/featureextractor/gstmetaalgorithmfeatures.h>
#include <gst/cuda/of/gstmetaopticalflow.h>
#include <gst/gst.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <gst/cuda/nvcodec/gstcudabasetransform.h>
#include <gst/cuda/nvcodec/gstcudacontext.h>
#include <gst/cuda/nvcodec/gstcudaloader.h>
#include <gst/cuda/nvcodec/gstcudamemory.h>
#include <gst/cuda/nvcodec/gstcudanvrtc.h>
#include <gst/cuda/nvcodec/gstcudautils.h>

/*
 * Just some setup for the GStreamer debug logger.
 *
 * - J.O.
 */
GST_DEBUG_CATEGORY_STATIC(gst_cuda_feature_extractor_debug);
#define GST_CAT_DEFAULT gst_cuda_feature_extractor_debug

#define GST_CUDA_FEATURE_EXTRACTOR(obj)        \
    (G_TYPE_CHECK_INSTANCE_CAST(               \
        (obj),                                 \
        gst_cuda_feature_extractor_get_type(), \
        GstCudaFeatureExtractor))

#define gst_cuda_feature_extractor_parent_class parent_class

#ifndef GST_CUDA_FEATURE_EXTRACTOR_KERNEL_SOURCE_PATH
#define GST_CUDA_FEATURE_EXTRACTOR_KERNEL_SOURCE_PATH \
    "./cudafeatureextractorkernels.cu"
#endif

#define GST_CUDA_FEATURE_EXTRACTOR_KERNEL "gst_cuda_feature_extractor_kernel"
#define GST_CUDA_FEATURE_CONSOLIDATION_KERNEL \
    "gst_cuda_feature_consolidation_kernel"

/****************************** Static Variables ******************************/

/**
 * \brief The maximum number of threads per block for CUDA kernels.
 *
 * \notes The dimensions used for launching a CUDA kernel must be setup so that
 * the X, Y and Z dimensions multiplied together do not exceed the 1024 threads
 * per block.  Otherwise CUDA will silently reject any attempts to run the CUDA
 * kernel.
 *
 * \notes Unfortunately, this is an issue that NVCC does not pick up on, nor
 * warn us about, so we have to make certain that this won't happen by
 * ourselves.
 */
static const guint32 cuda_max_threads_per_block = 1024u;

/**
 * \brief The default setting for the device-id property.
 *
 * \notes The default setting will result in the plugin either using the GPU
 * used by a preceeding plugin in the pipeline, or use the first listed GPU
 * installed into the system.
 */
static const gint default_device_id = -1;

/**
 * \brief The default setting for the enable-debug property.
 */
static const gboolean default_enable_debug = FALSE;

/**
 * \brief The default setting for the features-matrix-height property.
 */
static const guint32 default_features_matrix_height = 20u;

/**
 * \brief The default setting for the features-matrix-width property.
 */
static const guint32 default_features_matrix_width = 20u;

/**
 * \brief The default setting for the kernel-source-location property.
 *
 * \notes By default, it will look for CUDA kernels source file via the
 * following path: `$PREFIX/share/CUDA/cudafeatureextractor.cu`.
 */
static const gchar *default_kernel_source_location
    = GST_CUDA_FEATURE_EXTRACTOR_KERNEL_SOURCE_PATH;

/**
 * \brief The default setting for the magnitude-quadrant-threshold-squared
 * property.
 *
 * \notes It is still unknown why this value was chosen specifically for the
 * X0-To-X1, X1-To-X0, Y0-To-Y1 and Y1-To-Y0 magnitude features. However, as
 * this is the value chosen for V1, we are currently sticking to it for V2.
 */
static const gfloat default_magnitude_quadrant_threshold_squared = 2.25f;

/**
 * \brief The default setting for the motion-threshold-squared
 * property.
 *
 * \notes It is still unknown why this value was chosen specifically for the
 * Count feature. However, as this is the value chosen for V1, we are currently
 * sticking to it for V2.
 */
static const gfloat default_motion_threshold_squared = 4.0f;

/**
 * \brief The maximum multiplier for the features matrix dimensions prior to
 * being accumulated down to the requested features matrix dimensions.
 *
 * \notes From memory, the reason that the value of 400 was considered was that
 * it would support 8K footage for a 20x20 features matrix.
 */
static const guint32 feature_grid_dimensions_multiplier_max = 400u;

/**
 * \brief Small test kernel to confirm that NVRTC is loaded/working.
 */
static const gchar *nvrtc_test_source = "__global__ void test_kernel(void){}";

/**
 * \brief Anonymous enumeration containing the list of properties available for
 * the GstCudaFeatureExtractor GObject type this module defines.
 */
enum
{
    /**
     * \brief ID number for the GPU Device ID property.
     */
    PROP_DEVICE_ID = 1,

    /**
     * ID number for the enable-debug flag property.
     */
    PROP_ENABLE_DEBUG,

    /**
     * ID number for the features-matrix-height property.
     */
    PROP_FEATURES_MATRIX_HEIGHT,

    /**
     * ID number for the features-matrix-height property.
     */
    PROP_FEATURES_MATRIX_WIDTH,

    /**
     * ID number for the kernel-source-location property.
     */
    PROP_KERNEL_SOURCE_LOCATION,

    /**
     * ID number for the Magnitude Quadrant Threshold Squared property.
     */
    PROP_MAGNITUDE_QUADRANT_THRESHOLD_SQUARED,

    /**
     * ID number for the Motion Threshold Squared property.
     */
    PROP_MOTION_THRESHOLD_SQUARED,

    /**
     * \brief Number of property ID numbers in this enum.
     */
    N_PROPERTIES
};

/*
 * In a manner that's pretty consistent with C-style modules (I.E. one module
 * per C file), we store the properties as a static variable for the file.
 * There is currently no other way to do this in C that would let us store
 * variables like this at the class level.
 *
 * - J.O.
 */

/**
 * An array of GParamSpec instances representing the parameter specifications
 * for the parameters installed on the GstCudaFeatureExtractor GObject type.
 */
static GParamSpec *properties[N_PROPERTIES] = {
    NULL,
};

/*
 * Okay, we have two pad templates here. One for the sink (input) pad and the
 * other for the source (output) pad.
 *
 * Usually, we would be able to take advantage of pass-through mode, due to
 * identical caps. However, since we need to modify the metadata for the
 * buffers we received, we cannot use pass-through mode for this.
 *
 * - J.O.
 */

/**
 * \brief Sink pad template for the GstCudaFeatureExtractor GObject type.
 *
 * \notes The capability restrictions for this sink pad require device (GPU)
 * memory buffers for raw video data. As a result, this GStreamer element is
 * expected to be linked immediately after the nvh264dec GStreamer element or
 * another element that can support buffers with the CUDAMemory memory type.
 */
static GstStaticPadTemplate gst_cuda_feature_extractor_sink_template
    = GST_STATIC_PAD_TEMPLATE(
        "sink",
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw(memory:CUDAMemory)"));

/**
 * \brief Source pad template for the GstCudaFeatureExtractor GObject type.
 *
 * \notes The capability restrictions for this source pad require outputting
 * device (GPU) memory buffers for raw video data. This was done because this
 * GStreamer element is not expected to modify the video data; it will simply
 * be attaching metadata to the buffer. As such, the capabilities for the
 * source pad must match the capabilities for the sink pad.
 */
static GstStaticPadTemplate gst_cuda_feature_extractor_src_template
    = GST_STATIC_PAD_TEMPLATE(
        "src",
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw(memory:CUDAMemory)"));

/************************** Type/Struct Definitions ***************************/

/**
 * \brief A structure for representing an allocation of 2D pitched memory
 * within GPU memory.
 *
 * \notes It may be noticed that this structure is also redefined within the
 * CUDA kernel source-code as well. This is due to a limitation with using
 * `#include` due to how NVRTC has been setup for the NVCodec plugins.
 *
 * \notes As a result, it is heavily recommneded that this structure should NOT
 * BE TOUCHED under any circumstances; otherwise it could lead to
 * hard-to-detect faults with the CUDA kernels.
 */
typedef struct _CUDA2DPitchedArray
{
    /**
     * \brief The pointer to the device (GPU) memory.
     */
    void *device_ptr;

    /**
     * \brief The pitch for the allocated 2D memory space.
     *
     * \details This is the pitch for the allocated 2D memory space. This will
     * usually be either the same size as the width of the memory space
     * multiplied by the size of the data-type, or greater, depending on the
     * pitch chosen by CUDA.
     */
    gsize pitch;

    /**
     * \brief The width for the allocated 2D memory space in bytes.
     */
    gsize width;

    /**
     * \brief The height for the allocated 2D memory space in bytes.
     */
    gsize height;

    /**
     * \brief The size of the data-type for each element within the 2D memory
     * space.
     */
    gsize elem_size;
} CUDA2DPitchedArray;

/*
 * \brief The structure for the GstCudaFeatureExtractor GStreamer element type
 * containing the public instance data.
 */
typedef struct _GstCudaFeatureExtractor
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
    GstCudaBaseTransform parent;

    /********************************* Public *********************************/

    /**
     * \brief A flag that determines if certain debugging features are enabled
     * in order to find any bugs within the plugin.
     */
    gboolean enable_debug;

    /**
     * \brief The number of rows in the features matrix extracted by the
     * plugin.
     */
    guint32 features_matrix_height;

    /**
     * \brief The number of columns in the features matrix extracted by the
     * plugin.
     */
    guint32 features_matrix_width;

    /**
     * \brief The path to the file containing the source code for the feature
     * extractor & consolidator CUDA kernels.
     */
    gchar *kernel_source_location;

    /**
     * \brief The threshold for the X0ToX1Magnitude, X1ToX0Magnitude,
     * Y0ToY1Magnitude and Y1ToY0Magnitude features.
     *
     * \details This represents the threshold that must be exceeded for the
     * squared vector value before it will qualify for the X0ToX1Magnitude,
     * X1ToX0Magnitude, Y0ToY1Magnitude and Y1ToY0Magnitude features. In
     * particular:
     *   - If X ^ 2 exceeds the threshold, and the original X value is
     *   positive, its value is added to the X0ToX1Magnitude feature.
     *   - If Y ^ 2 exceeds the threshold, and the original Y value is
     *   positive, its value is added to the Y0ToY1Magnitude feature.
     *   - If X ^ 2 exceeds the threshold, and the original X value is
     *   negative, its value is added to the X1ToX0Magnitude feature.
     *   - If Y ^ 2 exceeds the threshold, and the original Y value is
     *   negative, its value is added to the Y1ToY0Magnitude feature.
     */
    gfloat magnitude_quadrant_threshold_squared;

    /**
     * \brief The threshold for the Count feature.
     *
     * \details This represents the threshold that must be exceeded for the
     * squared distance value, calculated from the sum of both the X and Y axis
     * values squared, in order for the optical flow motion vector to quality
     * for the Count feature.
     */
    gfloat motion_threshold_squared;
} GstCudaFeatureExtractor;

/*
 * \brief The structure for the GstCudaFeatureExtractor GStreamer element type
 * containing the private instance data.
 */
typedef struct _GstCudaFeatureExtractorPrivate
{
    /******************************** Private *********************************/

    /**
     * \brief The run-time compiled CUDA kernel module containing the
     * feature-extractor & feature-consolidator kernels.
     */
    CUmodule cuda_module;

    /**
     * \brief The number of frames that have been processed by the feature
     * extractor plugin.
     */
    guint64 frame_num;

    /**
     * \brief A pointer to the CUDA kernel function for the feature-extractor.
     */
    CUfunction feature_extractor_kernel;

    /**
     * \brief A pointer to the CUDA kernel function for the
     * feature-consolidator.
     */
    CUfunction feature_consolidator_kernel;

    /**
     * \brief The timestamp of the most recent frame that has been processed by
     * the feature extractor plugin.
     */
    GstClockTime frame_timestamp;
} GstCudaFeatureExtractorPrivate;

/**
 * \brief The structure for the GstCudaFeatureExtractor GStreamer element type
 * containing the class' data.
 *
 * \details This structure contains class-level data and the virtual function
 * table for itself and its parent classes. This structure will be shared
 * between all instances of the GstCudaFeatureExtractor type.
 */
typedef struct _GstCudaFeatureExtractorClass
{
    /********************************** Base **********************************/

    /**
     * \brief The parent class' structure.
     *
     * \details This is the structure for the parent class, containing the
     * virtual function table and class-level data for the parent class. When a
     * pointer to an instance of this class is cast to an instance to the
     * parent class or any other classes higher up in the class hierarchy, only
     * the variables and virtual functions available to that class will be
     * available to be modified or used.
     *
     * \notes As per above, this relies on a bit of trickery regarding how C
     * stores its data structures in memory. The order that the structures are
     * defined here are the order they will be stored in memory by C. That
     * allows us to "cheat" by casting a pointer to this structure to a pointer
     * of the parent structure(s); thereby giving us an inheritance-like
     * nature to these structures.
     */
    GstCudaBaseTransformClass parent_class;
} GstCudaFeatureExtractorClass;

/**
 * \brief Exception representing errors utilising GStreamer's CUDA
 * methods/data-types.
 *
 * \details This exception represents serveral potential errors that can occur
 * during the usage of the GstCudaContext data-type, or other possible GStreamer
 * CUDA methods/data-types.
 */
class GstCudaException : public std::logic_error
{
    public:
    /************************ Public Member Functions *************************/

    /**
     * \brief Default constructor.
     *
     * \details Constructs an instance of this exception with the given
     * message.
     */
    explicit GstCudaException(const std::string &what_arg)
        : std::logic_error(what_arg)
    {
    }

    /**
     * \brief Default constructor.
     *
     * \details Constructs an instance of this exception with the given
     * message.
     */
    explicit GstCudaException(const char *what_arg) : std::logic_error(what_arg)
    {
    }

    /**
     * \brief Copy constructor.
     *
     * \details Constructs an instance of this exception with the message from
     * another instance of this exception.
     */
    GstCudaException(const GstCudaException &other) noexcept
        : GstCudaException(other.what())
    {
    }

    /**
     * \brief Returns the instance's message.
     *
     * \returns The instance's message.
     */
    const char *what() const noexcept override
    {
        return std::logic_error::what();
    }
};

/**
 * \brief A structure containing the threshold values needed for feature
 * extraction.
 *
 * \notes It may be noticed that this structure is also redefined within the
 * CUDA kernel source-code as well. This is due to a limitation with using
 * `#include` due to how NVRTC has been setup for the NVCodec plugins.
 *
 * \notes As a result, it is heavily recommneded that this structure should NOT
 * BE TOUCHED under any circumstances; otherwise it could lead to
 * hard-to-detect faults with the CUDA kernels.
 */
typedef struct _MotionThresholds
{
    /**
     * \brief The threshold for the Count feature.
     *
     * \details This represents the threshold that must be exceeded for the
     * squared distance value, calculated from the sum of both the X and Y axis
     * values squared, in order for the optical flow motion vector to quality
     * for the Count feature.
     */
    float motion_threshold_squared;

    /**
     * \brief The threshold for the X0ToX1Magnitude, X1ToX0Magnitude,
     * Y0ToY1Magnitude and Y1ToY0Magnitude features.
     *
     * \details This represents the threshold that must be exceeded for the
     * squared vector value before it will qualify for the X0ToX1Magnitude,
     * X1ToX0Magnitude, Y0ToY1Magnitude and Y1ToY0Magnitude features. In
     * particular:
     *   - If X ^ 2 exceeds the threshold, and the original X value is
     *   positive, its value is added to the X0ToX1Magnitude feature.
     *   - If Y ^ 2 exceeds the threshold, and the original Y value is
     *   positive, its value is added to the Y0ToY1Magnitude feature.
     *   - If X ^ 2 exceeds the threshold, and the original X value is
     *   negative, its value is added to the X1ToX0Magnitude feature.
     *   - If Y ^ 2 exceeds the threshold, and the original Y value is
     *   negative, its value is added to the Y1ToY0Magnitude feature.
     */
    float magnitude_quadrant_threshold_squared;
} MotionThresholds;

/**
 * \brief A structure containing the feature values extracted from the optical
 * flow motion vector data.
 *
 * \notes It may be noticed that this structure is also redefined within the
 * CUDA kernel source-code as well. This is due to a limitation with using
 * `#include` due to how NVRTC has been setup for the NVCodec plugins.
 *
 * \notes As a result, it is heavily recommneded that this structure should NOT
 * BE TOUCHED under any circumstances; otherwise it could lead to
 * hard-to-detect faults with the CUDA kernels.
 */
typedef struct _MotionFeatures
{
    /**
     * \brief The total number of pixels contained within the features matrix
     * cell.
     */
    guint32 pixels;

    /**
     * \brief The number of optical flow vectors with a squared distance value
     * greater than the set threshold.
     */
    guint32 count;

    /**
     * \brief The cumulative absolute value of the positive X-planar values
     * within the optical flow vectors whose squared magnitude exceed a set
     * threshold.
     */
    float x0_to_x1_magnitude;

    /**
     * \brief The cumulative absolute value of the negative X-planar values
     * within the optical flow vectors whose squared magnitude exceed a set
     * threshold.
     */
    float x1_to_x0_magnitude;

    /**
     * \brief The cumulative absolute value of the positive Y-planar values
     * within the optical flow vectors whose squared magnitude exceed a set
     * threshold.
     */
    float y0_to_y1_magnitude;

    /**
     * \brief The cumulative absolute value of the negative Y-planar values
     * within the optical flow vectors whose squared magnitude exceed a set
     * threshold.
     */
    float y1_to_y0_magnitude;
} MotionFeatures;

/*************************** Function Declarations ****************************/

/**
 * \brief Calculate the multiplier value for the initial features matrix.
 *
 * \details Using the maximum number of threads per block (a CUDA limitation),
 * determine the size of the initial features matrix that will give us
 * dimensions allowing us to be able to successfully launch the CUDA kernel.
 * The initial features matrix will be later consolidated down to the desired
 * features matrix size.
 *
 * \param[in] optical_flow_matrix_width The number of optical flow motion
 * vector pairs per row for the matrix.
 * \param[in] optical_flow_matrix_height The number of rows for the optical
 * flow matrix.
 * \param[in] optical_flow_matrix_width The number of feature sets per row for
 * the matrix.
 * \param[in] optical_flow_matrix_height The number of rows for the features
 * matrix.
 */
static gsize gst_cuda_feature_extractor_calculate_dimensions_multiplier(
    const gsize optical_flow_matrix_width,
    const gsize optical_flow_matrix_height,
    const gsize features_matrix_width,
    const gsize features_matrix_height);

/**
 * \brief Disposal method for the GstCudaFeatureExtractor GObject type.
 *
 * \details Upon the destruction of an instance of the GstCudaFeatureExtractor
 * GObject, this method is called to clean-up the pointers and handles to the
 * run-time compiled CUDA kernels module.
 *
 * \param[in,out] object A GstCudaFeatureExtractor GObject instance to release
 * all held resources from.
 */
static void gst_cuda_feature_extractor_dispose(GObject *gobject);

/**
 * \brief Extracts features from optical flow metadata.
 *
 * \details Using the loaded feature-extractor and feature-consolidator
 * kernels, a set of 6 features are extracted from the optical flow matrix
 * stored in the optical flow metadata. The features are copied from GPU to
 * host memory, then stored within a CUDAFeaturesMatrix GObject instance to be
 * stored within a GstMetaAlgorithmFeatures metadata instance.
 *
 * \param[in] self A GstCudaFeatureExtractor GObject instance to get various
 * parameters and handles needed to perform the feature extraction procedure.
 * \param[in] optical_flow_metadata The GstMetaOpticalFlow instance to extract
 * the optical flow matrix from.
 *
 * \returns A reference to a CUDAFeaturesMatrix GObject instance.
 * Alternatively, NULL will be returned if an error occurs during the
 * feature-extraction/feature-consolidation procedures.
 */
static CUDAFeaturesMatrix *gst_cuda_feature_extractor_extract_features(
    GstCudaFeatureExtractor *self,
    const GstMetaOpticalFlow *optical_flow_metadata);

/**
 * \brief Wrapper around gst_cuda_feature_extractor_get_instance_private.
 *
 * \details Wraps around gst_cuda_feature_extractor_get_instance_private in
 * order to return a pointer to a GstCudaFeatureExtractorPrivate instance.
 *
 * \param[in] self A GstCudaFeatureExtractor GObject instance to get the
 * property from.
 *
 * \returns A pointer to the GstCudaFeatureExtractorPrivate instance for the
 * given GstCudaFeatureExtractor instance.
 */
static GstCudaFeatureExtractorPrivate *
gst_cuda_feature_extractor_get_instance_private_typesafe(
    GstCudaFeatureExtractor *self);

/**
 * \brief Property getter for instances of the GstCudaFeatureExtractor GObject.
 *
 * \param[in] gobject A GstCudaFeatureExtractor GObject instance to get the
 * property from.
 * \param[in] prop_id The ID number for the property to get. This should
 * correspond to an entry in the enum defined in this file.
 * \param[out] value The GValue instance (generic value container) that will
 * contain the value of the property after being converted into the generic
 * GValue type.
 * \param[in] pspec The property specification instance for the property that
 * we are getting. This property specification should exist on the
 * GstCudaFeatureExtractorClass GObjectClass structure.
 *
 * \notes Any property ID that does not exist on the
 * GstCudaFeatureExtractorClass GClassObject will result in that property being
 * ignored by the getter.
 */
static void gst_cuda_feature_extractor_get_property(
    GObject *gobject,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec);

/**
 * \brief Debugging method to output features to the filesystem as JSON files.
 *
 * \details If the enable-debug property is enabled, this method will dump the
 * features generated by the feature extractor to the filesystem by serialising
 * the features into JSON using RapidJSON. This helps users to confirm that the
 * feature extractor is working by examining the content of the JSON files.
 *
 * \param[in] self A GstCudaFeatureExtractor GObject instance to get properties
 * from to build the output file-name.
 * \param[in] frame The current frame being processed by the plugin.
 * \param[in] algorithm_features_metadata The metadata containing the features
 * matrix that will be output by the plugin.
 *
 * \returns TRUE if outputting the features matrix to a JSON file was
 * successful. FALSE otherwise.
 */
static gboolean gst_cuda_feature_extractor_output_features_json(
    GstCudaFeatureExtractor *self,
    const GstVideoFrame *frame,
    const GstMetaAlgorithmFeatures *algorithm_features_metadata);

/**
 * \brief Debugging method to output motion vectors to the filesystem as binary
 * files.
 *
 * \details If the enable-debug property is enabled, this method will dump the
 * optical flow vectors generated by the optical flow plugin to the filesystem
 * by downloading the optical flow vectors matrix directly to host memory and
 * writting the contents of the optical flow vectors matrix into a file.
 *
 * \details For most optical flow algorithms implemented by OpenCV, this will
 * result in a binary file with each optical flow vector being represented by
 * a pair of 32-bit floating-point values. As a file is only 1D by nature, the
 * matrix will have to be reshaped to a 2D matrix when reading the optical flow
 * vectors back from the file.
 *
 * \param[in] self A GstCudaFeatureExtractor GObject instance to get properties
 * from to build the output file-name.
 * \param[in] frame The current frame being processed by the plugin.
 * \param[in] optical_flow_metadata The metadata containing the optical flow
 * matrix that will be output by the plugin.
 *
 * \returns TRUE if outputting the optical flow matrix to a binary file was
 * successful. FALSE otherwise.
 *
 * \notes To best view the resultant optical flow file, use the following
 * command: `od -t f4 -w8 -v <OPTICAL_FLOW_VECTORS_FILE>`. This will allow you
 * to see the values of the optical flow vector pairs in sequence.
 */
static gboolean gst_cuda_feature_extractor_output_motion_vectors(
    GstCudaFeatureExtractor *self,
    const GstVideoFrame *frame,
    const GstMetaOpticalFlow *optical_flow_metadata);

/**
 * \brief Property setter for instances of the GstCudaFeatureExtractor GObject.
 *
 * \details Sets properties on an instance of the GstCudaFeatureExtractor
 * GObject.
 *
 * \details Any property ID that does not exist on the
 * GstCudaFeatureExtractorClass GClassObject will result in that property being
 * ignored by the setter.
 *
 * \param[in,out] gobject A GstCudaFeatureExtractor GObject instance to set the
 * property on.
 * \param[in] prop_id The ID number for the property to set. This should
 * correspond to an entry in the enum defined in this file.
 * \param[in] value The GValue instance (generic value container) that will be
 * convered to the correct value type to set the property with.
 * \param[in] pspec The property specification instance for the property that
 * we are setting. This property specification should exist on the
 * GstCudaFeatureExtractorClass GObjectClass structure.
 */
static void gst_cuda_feature_extractor_set_property(
    GObject *gobject,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec);

/**
 * \brief Sets up the element to begin processing.
 *
 * \details This method sets up the element for processing by calling the
 * parent class' implementation of this method. Then the number of frames
 * processed is cleared and set to zero.
 *
 * \param[in,out] trans A GstCudaFeatureExtractor GObject instance to setup for
 * processing.
 *
 * \returns TRUE if setup was successful. False if errors occurred.
 */
static gboolean gst_cuda_feature_extractor_start(GstBaseTransform *trans);

/**
 * \brief Cleans up the element to stop processing.
 *
 * \details This method cleans up the element by unloading the run-time
 * compiled CUDA kernel module, and clearing the pointers to the loaded CUDA
 * kernels. finally, the parent class' implementation of the method is called.
 *
 * \details After this method is called, no more processing is expected to be
 * performed by the element until it has been transitioned back into the
 * PLAYING state.
 *
 * \param[in,out] trans A GstCudaFeatureExtractor GObject instance to clean-up
 * and stop processing on.
 *
 * \returns TRUE if clean-up was successful. False if errors occurred.
 */
static gboolean gst_cuda_feature_extractor_stop(GstBaseTransform *trans);

/**
 * \brief Analyses and extracts features from the optical flow metadata
 * attached to the current buffer; attaching the features as metadata to the
 * output buffer.
 *
 * \details This method uses a standalone library to calculate the features for
 * a frame, given the calculated optical flow vectors. The optical flow
 * vectors, attached as metadata to the frame's buffer, is extracted, and the
 * pointer to the OpenCV GPU matrix is then passed to the standalone feature
 * extractor library. The resultant features are returned to this plugin, which
 * are then attached as metadata to the output buffer and passed to the next
 * attached element.
 *
 * \param[in] trans A GstCudaFeatureExtractor GObject instance.
 * \param[in] inbuf A pointer to the buffer that the GstCudaFeatureExtractor
 * element has received on its sink pad.
 * \param[in,out] outbuf A pointer to the buffer that the
 * GstCudaFeatureExtractor element will output on its source pad.
 *
 * \returns GST_FLOW_OK if no errors occurred. GST_FLOW_ERROR if an error
 * occurred during the extraction of features from the optical flow vectors.
 */
static GstFlowReturn gst_cuda_feature_extractor_transform_frame(
    GstCudaBaseTransform *filter,
    GstVideoFrame *in_frame,
    GstCudaMemory *in_cuda_mem,
    GstVideoFrame *out_frame,
    GstCudaMemory *out_cuda_mem);

/**
 * \brief Opens a file for writting formatted metadata from the current buffer
 * into.
 *
 * \details This function opens (and creates, if not currently existing) a file
 * within the current working directory for outputting metadata, such as the
 * features or optical flow data, into for debugging purposes.
 *
 * \param[in] self A GstCudaFeatureExtractor GObject instance to get properties
 * from to build the output file-name.
 * \param[in] frame The current frame being processed by the plugin.
 * \param[in] file_type The file-type of the output metadata file.
 *
 * \returns An open std::ofstream instance pointing to the output metadata file.
 *
 * \exception std::runtime_error If an error occurs when attempting to open the
 * output metadata file.
 */
static std::ofstream open_output_metadata_file(
    GstCudaFeatureExtractor *self,
    const GstVideoFrame *frame,
    const std::string &file_type);

/************************** GObject Type Definitions **************************/

/*
 * Okay, so this is the main GObject meta-programming macro for defining the
 * type. This macro generates a bunch of things; including the
 * gst_cuda_feature_extractor_get_type function, which defines the actual GObject
 * type during runtime.
 *
 * - J.O.
 */
G_DEFINE_TYPE_WITH_PRIVATE(
    GstCudaFeatureExtractor,
    gst_cuda_feature_extractor,
    GST_TYPE_CUDA_BASE_TRANSFORM)

/**************************** Function Definitions ****************************/

static inline guint calculate_dimension(guint size, guint divisor)
{
    return ((size + divisor - 1) / (divisor));
}

static gsize gst_cuda_feature_extractor_calculate_dimensions_multiplier(
    const gsize optical_flow_matrix_width,
    const gsize optical_flow_matrix_height,
    const gsize features_matrix_width,
    const gsize features_matrix_height)
{
    gsize dimensions_multiplier = 1;

    for(gsize ii = 1; ii < feature_grid_dimensions_multiplier_max; ii++)
    {
        /*
         * This is typical block dimension calculation that we use. Since it is
         * not always guaranteed that we will get evenly divisible grid sizes,
         * this is what we will use to determine the total number of threads in
         * each block of the feature grid.
         *
         * - J.O.
         */
        guint32 block_dimension_x = calculate_dimension(
            optical_flow_matrix_width, features_matrix_width * ii);
        guint32 block_dimension_y = calculate_dimension(
            optical_flow_matrix_height, features_matrix_height * ii);

        /*
         * The aim of this function is to find a feature grid size that remains
         * equal to, or less than the maximum allowed number of threads in a
         * CUDA kernel block; currently 1024. All other feature grid sizes
         * should not be considered.
         *
         * - J.O.
         */
        if((block_dimension_x * block_dimension_y)
           <= cuda_max_threads_per_block)
        {
            /*
             * This is a bit of a horrible way to do it, but I end up saving
             * the first multiplier value that managed to be under 1024 threads
             * per block. This is used if we cannot find any feature grid
             * dimensions that evenly divide into the frame's dimensions.
             *
             * - J.O.
             */
            if(dimensions_multiplier == 1 && dimensions_multiplier != ii)
            {
                dimensions_multiplier = ii;
            }

            /*
             * If we find a feature grid size that is evenly divisible, and
             * results in less than 1024 threads per block, then we have our
             * answer here. We save the answer and break out of the for-loop.
             *
             * - J.O.
             */
            if((optical_flow_matrix_width % (features_matrix_width * ii) == 0)
               && (optical_flow_matrix_height % (features_matrix_height * ii)
                   == 0))
            {
                dimensions_multiplier = ii;
                break;
            }
        }

        /*
         * If the next feature grid size would result in the dimensions for the
         * feature grid exceeding the dimensions for the frame, then there's no
         * point continuing any further. We need to give up and use the first
         * multiplier value that calculated to a block size of less than 1024
         * threads.
         *
         * - J.O.
         */
        if((optical_flow_matrix_width <= features_matrix_width * (ii + 1))
           || (optical_flow_matrix_height <= features_matrix_height * (ii + 1)))
        {
            break;
        }
    }

    return dimensions_multiplier;
}

/**
 * \brief Initialisation function for the GstCudaFeatureExtractorClass
 * GObjectClass type.
 *
 * \details This is a class initialisation function that is called the first
 * time a GstCudaFeatureExtractor instance is allocated.
 *
 * \details The class structure is setup with the necessary properties, virtual
 * method overrides, element pads and element metadata for a GObject class
 * derived from one of GStreamer's base element types.
 *
 * \param[in,out] klass The instance of the GstCudaFeatureExtractorClass
 * GObjectClass structure.
 */
static void
gst_cuda_feature_extractor_class_init(GstCudaFeatureExtractorClass *klass)
{
    /*
     * Okay, so this is one of the main two initialisation functions. This will
     * get called the first time an instance of the GstCudaFeatureExtractor
     * GObject type is created (typically via g_object_new). This will let us
     * setup the GstCudaFeatureExtractorClass GObjectClass structure for all
     * GstCudaFeatureExtractor instances.
     *
     * This allows us to setup properties, virtual function pointers (though
     * only our base classes have them for now) and potentially even
     * class-level variables.
     *
     * - J.O.
     */
    GObjectClass *gobject_class = NULL;
    GstElementClass *gstelement_class = NULL;
    GstBaseTransformClass *gstbasetransform_class = NULL;
    GstCudaBaseTransformClass *gstcudabasetransform_class = NULL;

    gobject_class = G_OBJECT_CLASS(klass);
    gstelement_class = GST_ELEMENT_CLASS(klass);
    gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);
    gstcudabasetransform_class = GST_CUDA_BASE_TRANSFORM_CLASS(klass);

    gobject_class->dispose
        = GST_DEBUG_FUNCPTR(gst_cuda_feature_extractor_dispose);
    gobject_class->set_property
        = GST_DEBUG_FUNCPTR(gst_cuda_feature_extractor_set_property);
    gobject_class->get_property
        = GST_DEBUG_FUNCPTR(gst_cuda_feature_extractor_get_property);

    properties[PROP_DEVICE_ID] = g_param_spec_int(
        "cuda-device-id",
        "Cuda Device ID",
        "Set the GPU device to use for operations (-1 = auto)",
        -1,
        G_MAXINT,
        default_device_id,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_ENABLE_DEBUG] = g_param_spec_boolean(
        "enable-debug",
        "Enable Debug",
        "Enables debug output for the plugin. Motion vector and feature files "
        "will be output into the current working directory.",
        default_enable_debug,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    properties[PROP_FEATURES_MATRIX_HEIGHT] = g_param_spec_uint(
        "features-matrix-height",
        "Features Matrix Height",
        "The number of rows for the features matrix being output by the "
        "plugin.",
        0,
        G_MAXUINT32,
        default_features_matrix_height,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FEATURES_MATRIX_WIDTH] = g_param_spec_uint(
        "features-matrix-width",
        "Features Matrix Width",
        "The number of columns for the features matrix being output by the "
        "plugin.",
        0,
        G_MAXUINT32,
        default_features_matrix_width,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_KERNEL_SOURCE_LOCATION] = g_param_spec_string(
        "kernel-source-location",
        "Kernel Source Location",
        "Specifies the filepath for the feature extractor kernel source "
        "compiled by NVRTC during runtime.",
        default_kernel_source_location,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_MAGNITUDE_QUADRANT_THRESHOLD_SQUARED] = g_param_spec_float(
        "magnitude-quadrant-threshold-squared",
        "Magnitude Quadrant Threshold Squared",
        "Modifies the threshold value for the X0ToX1Magnitude, "
        "X1ToX0Magnitude, Y0ToY1Magnitude and Y1ToY0Magnitude features.",
        0.0f,
        G_MAXFLOAT,
        default_magnitude_quadrant_threshold_squared,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_MOTION_THRESHOLD_SQUARED] = g_param_spec_float(
        "motion-threshold-squared",
        "Motion Threshold Squared",
        "Modifies the threshold value for the Count feature.",
        0.0f,
        G_MAXFLOAT,
        default_motion_threshold_squared,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    g_object_class_install_properties(gobject_class, N_PROPERTIES, properties);

    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_cuda_feature_extractor_sink_template));
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_cuda_feature_extractor_src_template));

    gst_element_class_set_metadata(
        gstelement_class,
        "CUDA Optical Flow Feature Extractor",
        "Filter/Video/Hardware",
        "Processes GPU-hosted optical flow metadata to generate features, then "
        "stores the features as buffer metadata.",
        "icetana");

    gstbasetransform_class->start
        = GST_DEBUG_FUNCPTR(gst_cuda_feature_extractor_start);
    gstbasetransform_class->stop
        = GST_DEBUG_FUNCPTR(gst_cuda_feature_extractor_stop);

    gstbasetransform_class->passthrough_on_same_caps = FALSE;
    gstbasetransform_class->transform_ip_on_passthrough = FALSE;

    gstcudabasetransform_class->transform_frame
        = GST_DEBUG_FUNCPTR(gst_cuda_feature_extractor_transform_frame);
}

static void gst_cuda_feature_extractor_dispose(GObject *gobject)
{
    GstCudaBaseTransform *filter = GST_CUDA_BASE_TRANSFORM(gobject);
    GstCudaFeatureExtractor *self = GST_CUDA_FEATURE_EXTRACTOR(gobject);
    GstCudaFeatureExtractorPrivate *self_private
        = gst_cuda_feature_extractor_get_instance_private_typesafe(self);

    if(filter->context != NULL)
    {
        if(gst_cuda_context_push(filter->context))
        {
            self_private->feature_consolidator_kernel = NULL;
            self_private->feature_extractor_kernel = NULL;

            if(self_private->cuda_module != NULL)
            {
                gst_cuda_result(CuModuleUnload(self_private->cuda_module));
                self_private->cuda_module = NULL;
            }

            gst_cuda_context_pop(NULL);
        }
    }

    if(G_OBJECT_CLASS(gst_cuda_feature_extractor_parent_class)->dispose != NULL)
    {
        G_OBJECT_CLASS(gst_cuda_feature_extractor_parent_class)
            ->dispose(gobject);
    }
}

static CUDAFeaturesMatrix *gst_cuda_feature_extractor_extract_features(
    GstCudaFeatureExtractor *self,
    const GstMetaOpticalFlow *optical_flow_metadata)
{
    GstCudaFeatureExtractorPrivate *self_private
        = gst_cuda_feature_extractor_get_instance_private_typesafe(self);

    CUDAFeaturesMatrix *features_matrix = NULL;

    const cv::cuda::GpuMat *optical_flow_matrix
        = optical_flow_metadata->optical_flow_vectors;
    const int optical_flow_vector_grid_size = optical_flow_metadata->optical_flow_vector_grid_size;

    const gsize features_matrix_width = self->features_matrix_width;
    const gsize features_matrix_height = self->features_matrix_height;
    const gsize features_matrix_pitch
        = self->features_matrix_width * sizeof(MotionFeatures);
    const gsize features_matrix_elem_size = sizeof(MotionFeatures);

    const gsize optical_flow_matrix_width = optical_flow_matrix->cols;
    const gsize optical_flow_matrix_height = optical_flow_matrix->rows;
    const gsize optical_flow_matrix_pitch = optical_flow_matrix->step;
    const gsize optical_flow_matrix_elem_size = optical_flow_matrix->elemSize();

    const size_t dimensions_multiplier
        = gst_cuda_feature_extractor_calculate_dimensions_multiplier(
            optical_flow_matrix_width * optical_flow_vector_grid_size,
            optical_flow_matrix_height * optical_flow_vector_grid_size,
            features_matrix_width,
            features_matrix_height);

    CUDA2DPitchedArray gpu_features_matrix = {
        NULL,
        features_matrix_pitch * dimensions_multiplier,
        features_matrix_width * features_matrix_elem_size
            * dimensions_multiplier,
        features_matrix_height * dimensions_multiplier,
        features_matrix_elem_size};
    CUDA2DPitchedArray consolidated_gpu_features_matrix
        = {
        NULL,
        features_matrix_pitch,
        features_matrix_width * features_matrix_elem_size,
        features_matrix_height,
        features_matrix_elem_size};
    CUDA2DPitchedArray gpu_optical_flow_matrix
        = {
        optical_flow_matrix->data,
        optical_flow_matrix_pitch,
        optical_flow_matrix_width * optical_flow_matrix_elem_size,
        optical_flow_matrix_height,
        optical_flow_matrix_elem_size};

    std::vector<MotionFeatures> host_features_matrix(
        features_matrix_width * features_matrix_height);

    MotionThresholds gpu_features_thresholds;
    gpu_features_thresholds.magnitude_quadrant_threshold_squared
        = self->magnitude_quadrant_threshold_squared;
    gpu_features_thresholds.motion_threshold_squared
        = self->motion_threshold_squared;

    try
    {
        guint original_grid_dimension_x
            = features_matrix_width * dimensions_multiplier;
        guint original_grid_dimension_y
            = features_matrix_height * dimensions_multiplier;

        guint original_block_dimension_x = calculate_dimension(
            (optical_flow_matrix_width * optical_flow_vector_grid_size), original_grid_dimension_x);
        guint original_block_dimension_y = calculate_dimension(
            (optical_flow_matrix_height * optical_flow_vector_grid_size), original_grid_dimension_y);

        if(!gst_cuda_result(CuMemAllocPitch(
               (CUdeviceptr *)&(gpu_features_matrix.device_ptr),
               &(gpu_features_matrix.pitch),
               gpu_features_matrix.width,
               gpu_features_matrix.height,
               16)))
        {
            throw GstCudaException(
                "Could not allocate GPU memory for the features matrix.");
        };

        gpointer feature_extractor_kernel_args[]
            = {&gpu_optical_flow_matrix,
               (gpointer)(&optical_flow_vector_grid_size),
               &gpu_features_thresholds,
               &gpu_features_matrix};

        if(!gst_cuda_result(CuLaunchKernel(
               self_private->feature_extractor_kernel,
               original_grid_dimension_x,
               original_grid_dimension_y,
               1,
               original_block_dimension_x,
               original_block_dimension_y,
               1,
               0,
               NULL,
               feature_extractor_kernel_args,
               NULL)))
        {
            throw GstCudaException(
                "Could not launch feature extractor CUDA kernel.");
        }

        guint consolidated_grid_dimension_x = features_matrix_width;
        guint consolidated_grid_dimension_y = features_matrix_height;

        guint consolidated_block_dimension_x = dimensions_multiplier;
        guint consolidated_block_dimension_y = dimensions_multiplier;

        if(!gst_cuda_result(CuMemAllocPitch(
               (CUdeviceptr *)&(consolidated_gpu_features_matrix.device_ptr),
               &(consolidated_gpu_features_matrix.pitch),
               consolidated_gpu_features_matrix.width,
               consolidated_gpu_features_matrix.height,
               16)))
        {
            throw GstCudaException(
                "Could not allocate GPU memory for the consolidated features "
                "matrix.");
        }

        gpointer feature_consolidator_kernel_args[]
            = {&gpu_features_matrix, &consolidated_gpu_features_matrix};

        if(!gst_cuda_result(CuLaunchKernel(
               self_private->feature_consolidator_kernel,
               consolidated_grid_dimension_x,
               consolidated_grid_dimension_y,
               1,
               consolidated_block_dimension_x,
               consolidated_block_dimension_y,
               1,
               0,
               NULL,
               feature_consolidator_kernel_args,
               NULL)))
        {
            throw GstCudaException(
                "Could not launch feature consolidator CUDA kernel.");
        }

        CUDA_MEMCPY2D feature_memcpy_args = {
            0,
        };

        feature_memcpy_args.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        feature_memcpy_args.srcDevice
            = (CUdeviceptr)(consolidated_gpu_features_matrix.device_ptr);
        feature_memcpy_args.srcPitch = consolidated_gpu_features_matrix.pitch;

        feature_memcpy_args.dstMemoryType = CU_MEMORYTYPE_HOST;
        feature_memcpy_args.dstHost = host_features_matrix.data();
        feature_memcpy_args.dstPitch
            = sizeof(MotionFeatures) * features_matrix_width;

        feature_memcpy_args.WidthInBytes
            = sizeof(MotionFeatures) * features_matrix_width;
        feature_memcpy_args.Height = features_matrix_height;

        if(!gst_cuda_result(CuMemcpy2D(&feature_memcpy_args)))
        {
            throw GstCudaException(
                "Could not copy features matrix to host memory.");
        }

        features_matrix = CUDA_FEATURES_MATRIX(g_object_new(
            CUDA_TYPE_FEATURES_MATRIX,
            // clang-format off
                "features-matrix-rows", self->features_matrix_height,
                "features-matrix-cols", self->features_matrix_width,
            // clang-format on
            NULL));

        for(gsize ii = 0; ii < features_matrix_height; ii++)
        {
            for(gsize jj = 0; jj < features_matrix_width; jj++)
            {
                CUDAFeaturesCell *features_cell
                    = cuda_features_matrix_at(features_matrix, jj, ii);

                g_object_set(
                    features_cell,
                    // clang-format off
                    "count", host_features_matrix[ii * features_matrix_width + jj].count,
                    "pixels", host_features_matrix[ii * features_matrix_width + jj].pixels,
                    "x0-to-x1-magnitude", host_features_matrix[ii * features_matrix_width + jj].x0_to_x1_magnitude,
                    "x1-to-x0-magnitude", host_features_matrix[ii * features_matrix_width + jj].x1_to_x0_magnitude,
                    "y0-to-y1-magnitude", host_features_matrix[ii * features_matrix_width + jj].y0_to_y1_magnitude,
                    "y1-to-y0-magnitude", host_features_matrix[ii * features_matrix_width + jj].y1_to_y0_magnitude,
                    // clang-format on
                    NULL);

                g_object_unref(features_cell);
            }
        }

        if(consolidated_gpu_features_matrix.device_ptr != NULL)
        {
            CuMemFree(
                (CUdeviceptr)(consolidated_gpu_features_matrix.device_ptr));
            consolidated_gpu_features_matrix.device_ptr = NULL;
        }

        if(gpu_features_matrix.device_ptr != NULL)
        {
            CuMemFree((CUdeviceptr)(gpu_features_matrix.device_ptr));
            gpu_features_matrix.device_ptr = NULL;
        }
    }
    catch(std::exception &ex)
    {
        GST_ERROR_OBJECT(self, "%s", ex.what());

        if(consolidated_gpu_features_matrix.device_ptr != NULL)
        {
            CuMemFree(
                (CUdeviceptr)(consolidated_gpu_features_matrix.device_ptr));
            consolidated_gpu_features_matrix.device_ptr = NULL;
        }

        if(gpu_features_matrix.device_ptr != NULL)
        {
            CuMemFree((CUdeviceptr)(gpu_features_matrix.device_ptr));
            gpu_features_matrix.device_ptr = NULL;
        }

        if(features_matrix != NULL)
        {
            g_object_unref(features_matrix);
            features_matrix = NULL;
        }
    }

    return features_matrix;
}

static GstCudaFeatureExtractorPrivate *
gst_cuda_feature_extractor_get_instance_private_typesafe(
    GstCudaFeatureExtractor *self)
{
    return static_cast<GstCudaFeatureExtractorPrivate *>(
        gst_cuda_feature_extractor_get_instance_private(self));
};

static void gst_cuda_feature_extractor_get_property(
    GObject *gobject,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec)
{
    GstCudaFeatureExtractor *gst_cuda_feature_extractor
        = GST_CUDA_FEATURE_EXTRACTOR(gobject);

    switch(prop_id)
    {
        /*
         * Okay, so this is where things get confusing. GValue is a universal
         * value types (of sorts) that GLib provides. In order to get and
         * set values for a GValue, we have to use these helper functions.
         *
         * See: https://developer.gnome.org/gobject/unstable/gobject-Generic-values.html
         *
         * -J.O.
         */
        case PROP_DEVICE_ID:
            g_value_set_int(
                value, gst_cuda_feature_extractor->parent.device_id);
            break;
        case PROP_ENABLE_DEBUG:
            g_value_set_boolean(
                value, gst_cuda_feature_extractor->enable_debug);
            break;
        case PROP_FEATURES_MATRIX_HEIGHT:
            g_value_set_uint(
                value, gst_cuda_feature_extractor->features_matrix_height);
            break;
        case PROP_FEATURES_MATRIX_WIDTH:
            g_value_set_uint(
                value, gst_cuda_feature_extractor->features_matrix_width);
            break;
        case PROP_KERNEL_SOURCE_LOCATION:
            g_value_set_string(
                value, gst_cuda_feature_extractor->kernel_source_location);
            break;
        case PROP_MAGNITUDE_QUADRANT_THRESHOLD_SQUARED:
            g_value_set_float(
                value,
                gst_cuda_feature_extractor
                    ->magnitude_quadrant_threshold_squared);
            break;
        case PROP_MOTION_THRESHOLD_SQUARED:
            g_value_set_float(
                value, gst_cuda_feature_extractor->motion_threshold_squared);
            break;
        default:
            g_assert_not_reached();
    }
}

/**
 * \brief Initialisation function for GstCudaFeatureExtractor instances.
 *
 * \details Upon allocation of a GstCudaFeatureExtractor instance, the object
 * structure for the instance is setup with the necessary property defaults.
 *
 * \param[in,out] self An instance of the GstCudaFeatureExtractor GObject type.
 */
static void gst_cuda_feature_extractor_init(GstCudaFeatureExtractor *self)
{
    /*
     * This is the other of the two main initialisation functions for the
     * GstCudaFeatureExtractor GObject type. This one is called each and every
     * time a new instance of the type is created. It sets up the flags for the
     * base class and the default values for the properties.
     *
     * - J.O.
     */
    GstBaseTransform *trans = GST_BASE_TRANSFORM(self);
    GstCudaFeatureExtractorPrivate *self_private
        = gst_cuda_feature_extractor_get_instance_private_typesafe(self);

    self->parent.device_id = default_device_id;

    self->enable_debug = default_enable_debug;
    self->features_matrix_height = default_features_matrix_height;
    self->features_matrix_width = default_features_matrix_width;
    self->kernel_source_location = g_strdup(default_kernel_source_location);
    self->magnitude_quadrant_threshold_squared
        = default_magnitude_quadrant_threshold_squared;
    self->motion_threshold_squared = default_motion_threshold_squared;

    self_private->cuda_module = NULL;
    self_private->feature_consolidator_kernel = NULL;
    self_private->feature_extractor_kernel = NULL;
    self_private->frame_num = 0;
    self_private->frame_timestamp = GST_CLOCK_TIME_NONE;

    gst_base_transform_set_in_place(trans, TRUE);
    gst_base_transform_set_gap_aware(trans, FALSE);
    gst_base_transform_set_passthrough(trans, FALSE);
    gst_base_transform_set_prefer_passthrough(trans, FALSE);
}

gboolean gst_cuda_feature_extractor_plugin_init(GstPlugin *plugin)
{
    /*
     * This is a strange little initialisation function that is exclusive to
     * GStreamer; this will otherwise not be seen in GObject-based programs.
     *
     * This is called the first, and only time, the plugin is initialised. This
     * lets us setup logging categories and register elements with the plugin
     * (we can have more than one element to a plugin).
     *
     * - J.O.
     */
    GST_DEBUG_CATEGORY_INIT(
        gst_cuda_feature_extractor_debug,
        "cudafeatureextractor",
        0,
        "CUDA Optical flow feature extractor");

    if(!gst_cuda_load_library())
    {
        return FALSE;
    }

    if(!gst_nvrtc_load_library())
    {
        return FALSE;
    }

    gchar *test_ptx = gst_cuda_nvrtc_compile(nvrtc_test_source);

    if(test_ptx == NULL)
    {
        return FALSE;
    }
    g_free(test_ptx);

    return gst_element_register(
        plugin,
        "cudafeatureextractor",
        GST_RANK_NONE,
        gst_cuda_feature_extractor_get_type());
}

static gboolean gst_cuda_feature_extractor_output_features_json(
    GstCudaFeatureExtractor *self,
    const GstVideoFrame *frame,
    const GstMetaAlgorithmFeatures *algorithm_features_metadata)
{
    gboolean result = TRUE;

    try
    {
        if(algorithm_features_metadata == NULL)
        {
            throw std::invalid_argument(
                "The given pointer to the algorithm features metadata is a "
                "null pointer.");
        }

        if(algorithm_features_metadata->features == NULL)
        {
            throw std::invalid_argument(
                "The given pointer to the features within the algorithm "
                "features metadata is a null pointer.");
        }

        GstCudaFeatureExtractorPrivate *self_private
            = gst_cuda_feature_extractor_get_instance_private_typesafe(self);

        std::ofstream algorithm_features_file
            = open_output_metadata_file(self, frame, ".json");

        guint32 feature_grid_width = 0;
        guint32 feature_grid_height = 0;
        g_object_get(
            algorithm_features_metadata->features,
            "features-matrix-cols",
            &feature_grid_width,
            "features-matrix-rows",
            &feature_grid_height,
            NULL);

        rapidjson::Document document;
        document.SetObject();

        rapidjson::Document::AllocatorType &allocator = document.GetAllocator();
        document.AddMember(
            "Frame-Number",
            rapidjson::Value(self_private->frame_num + 1),
            allocator);
        document.AddMember(
            "Frame-Timestamp",
            rapidjson::Value(self_private->frame_timestamp),
            allocator);
        document.AddMember(
            "Number-Of-Features", rapidjson::Value(6), allocator);
        document.AddMember(
            "Feature-Grid-Width",
            rapidjson::Value(feature_grid_width),
            allocator);
        document.AddMember(
            "Feature-Grid-Height",
            rapidjson::Value(feature_grid_height),
            allocator);

        rapidjson::Value features(rapidjson::kObjectType);
        rapidjson::Value cell_pixels_count_array(rapidjson::kArrayType);
        rapidjson::Value motion_count_array(rapidjson::kArrayType);
        rapidjson::Value x0_to_x1_magnitude_array(rapidjson::kArrayType);
        rapidjson::Value x1_to_x0_magnitude_array(rapidjson::kArrayType);
        rapidjson::Value y0_to_y1_magnitude_array(rapidjson::kArrayType);
        rapidjson::Value y1_to_y0_magnitude_array(rapidjson::kArrayType);

        for(guint32 rows = 0; rows < feature_grid_height; rows++)
        {
            for(guint32 cols = 0; cols < feature_grid_width; cols++)
            {
                CUDAFeaturesCell *cell = cuda_features_matrix_at(
                    algorithm_features_metadata->features, cols, rows);

                if(cell != NULL)
                {
                    guint count = 0;
                    guint pixels = 0;
                    gfloat x0_to_x1_magnitude = 0.0f;
                    gfloat x1_to_x0_magnitude = 0.0f;
                    gfloat y0_to_y1_magnitude = 0.0f;
                    gfloat y1_to_y0_magnitude = 0.0f;

                    g_object_get(
                        cell,
                        "count",
                        &count,
                        "pixels",
                        &pixels,
                        "x0-to-x1-magnitude",
                        &x0_to_x1_magnitude,
                        "x1-to-x0-magnitude",
                        &x1_to_x0_magnitude,
                        "y0-to-y1-magnitude",
                        &y0_to_y1_magnitude,
                        "y1-to-y0-magnitude",
                        &y1_to_y0_magnitude,
                        NULL);

                    cell_pixels_count_array.PushBack(
                        rapidjson::Value(pixels), allocator);
                    motion_count_array.PushBack(
                        rapidjson::Value(count), allocator);
                    x0_to_x1_magnitude_array.PushBack(
                        rapidjson::Value(x0_to_x1_magnitude), allocator);
                    x1_to_x0_magnitude_array.PushBack(
                        rapidjson::Value(x1_to_x0_magnitude), allocator);
                    y0_to_y1_magnitude_array.PushBack(
                        rapidjson::Value(y0_to_y1_magnitude), allocator);
                    y1_to_y0_magnitude_array.PushBack(
                        rapidjson::Value(y1_to_y0_magnitude), allocator);

                    gst_object_unref(cell);
                }
                else
                {
                    cell_pixels_count_array.PushBack(
                        rapidjson::Value(0), allocator);
                    motion_count_array.PushBack(rapidjson::Value(0), allocator);
                    x0_to_x1_magnitude_array.PushBack(
                        rapidjson::Value(0.0f), allocator);
                    x1_to_x0_magnitude_array.PushBack(
                        rapidjson::Value(0.0f), allocator);
                    y0_to_y1_magnitude_array.PushBack(
                        rapidjson::Value(0.0f), allocator);
                    y1_to_y0_magnitude_array.PushBack(
                        rapidjson::Value(0.0f), allocator);
                }
            }
        }

        features.AddMember(
            "Cell-Pixels-Count", cell_pixels_count_array, allocator);
        features.AddMember("Motion-Count", motion_count_array, allocator);
        features.AddMember(
            "X0-To-X1-Magnitude", x0_to_x1_magnitude_array, allocator);
        features.AddMember(
            "X1-To-X0-Magnitude", x1_to_x0_magnitude_array, allocator);
        features.AddMember(
            "Y0-To-Y1-Magnitude", y0_to_y1_magnitude_array, allocator);
        features.AddMember(
            "Y1-To-Y0-Magnitude", y1_to_y0_magnitude_array, allocator);

        document.AddMember("Features", features, allocator);

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);

        algorithm_features_file << buffer.GetString();
    }
    catch(std::invalid_argument &ex)
    {
        GST_ERROR_OBJECT(self, "Invalid argument error - %s", ex.what());
        result = FALSE;
    }
    catch(std::runtime_error &ex)
    {
        GST_ERROR_OBJECT(self, "Runtime error - %s", ex.what());
        result = FALSE;
    }
    catch(cv::Exception &ex)
    {
        GST_ERROR_OBJECT(self, "OpenCV error - %s", ex.what());
        result = FALSE;
    }
    catch(std::exception &ex)
    {
        GST_ERROR_OBJECT(self, "General error - %s", ex.what());
        result = FALSE;
    }

    return result;
}

static gboolean gst_cuda_feature_extractor_output_motion_vectors(
    GstCudaFeatureExtractor *self,
    const GstVideoFrame *frame,
    const GstMetaOpticalFlow *optical_flow_metadata)
{
    gboolean result = TRUE;

    try
    {
        if(optical_flow_metadata == NULL)
        {
            throw std::invalid_argument(
                "The given pointer to the optical flow metadata is a null "
                "pointer.");
        }

        if(optical_flow_metadata->optical_flow_vectors == NULL)
        {
            throw std::invalid_argument(
                "The given pointer to the GPU optical flow vectors within the "
                "optical flow metadata is a null pointer.");
        }

        std::ofstream motion_vectors_file
            = open_output_metadata_file(self, frame, ".mv");

        cv::Mat optical_flow_vectors;
        optical_flow_metadata->optical_flow_vectors->download(
            optical_flow_vectors);

        motion_vectors_file.write(
            reinterpret_cast<char *>(optical_flow_vectors.data),
            sizeof(float) * 2 * optical_flow_vectors.rows
                * optical_flow_vectors.cols);
    }
    catch(std::invalid_argument &ex)
    {
        GST_ERROR_OBJECT(self, "Invalid argument error - %s", ex.what());
        result = FALSE;
    }
    catch(std::runtime_error &ex)
    {
        GST_ERROR_OBJECT(self, "Runtime error - %s", ex.what());
        result = FALSE;
    }
    catch(cv::Exception &ex)
    {
        GST_ERROR_OBJECT(self, "OpenCV error - %s", ex.what());
        result = FALSE;
    }
    catch(std::exception &ex)
    {
        GST_ERROR_OBJECT(self, "General error - %s", ex.what());
        result = FALSE;
    }

    return result;
}

static void gst_cuda_feature_extractor_set_property(
    GObject *gobject,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec)
{
    GstCudaFeatureExtractor *gst_cuda_feature_extractor
        = GST_CUDA_FEATURE_EXTRACTOR(gobject);

    switch(prop_id)
    {
        /*
         * Typical property setter code for each of the below case statements.
         * Check to see if the property has changed. If it has, then set the
         * property's corresponding value in the GObject structure and notify
         * anyone who has attached listeners to that property.
         *
         * - J.O.
         */
        case PROP_DEVICE_ID:
            gst_cuda_feature_extractor->parent.device_id
                = g_value_get_int(value);
            break;
        case PROP_ENABLE_DEBUG:
            gst_cuda_feature_extractor->enable_debug
                = g_value_get_boolean(value);
            break;
        case PROP_FEATURES_MATRIX_HEIGHT:
            gst_cuda_feature_extractor->features_matrix_height
                = g_value_get_uint(value);
            break;
        case PROP_FEATURES_MATRIX_WIDTH:
            gst_cuda_feature_extractor->features_matrix_width
                = g_value_get_uint(value);
            break;
        case PROP_KERNEL_SOURCE_LOCATION:
            if(g_strcmp0(
                   gst_cuda_feature_extractor->kernel_source_location,
                   g_value_get_string(value)))
            {
                g_free(gst_cuda_feature_extractor->kernel_source_location);
                gst_cuda_feature_extractor->kernel_source_location
                    = g_strdup(g_value_get_string(value));
            }
            break;
        case PROP_MAGNITUDE_QUADRANT_THRESHOLD_SQUARED:
            gst_cuda_feature_extractor->magnitude_quadrant_threshold_squared
                = g_value_get_float(value);
            break;
        case PROP_MOTION_THRESHOLD_SQUARED:
            gst_cuda_feature_extractor->motion_threshold_squared
                = g_value_get_float(value);
            break;
        default:
            g_assert_not_reached();
    }
}

static gboolean gst_cuda_feature_extractor_start(GstBaseTransform *trans)
{
    GstCudaBaseTransform *filter = GST_CUDA_BASE_TRANSFORM(trans);
    GstCudaFeatureExtractor *self = GST_CUDA_FEATURE_EXTRACTOR(trans);
    GstCudaFeatureExtractorPrivate *self_private
        = gst_cuda_feature_extractor_get_instance_private_typesafe(self);

    gboolean result = TRUE;

    result = GST_BASE_TRANSFORM_CLASS(parent_class)->start(trans);

    if(result)
    {
        self_private->frame_num = 0;
        self_private->frame_timestamp = GST_CLOCK_TIME_NONE;

        if(gst_cuda_context_push(filter->context))
        {
            gchar *ptx = NULL;

            try
            {
                /*
                 * Another explanation:
                 *
                 * Until GCC version 8, the C++17 filesystem library within the STL was
                 * also included under a different namespace path
                 * (std::experimental::filesystem) compared to the actual specification
                 * (std::filesystem).
                 *
                 * As a result, we need a namespace alias in order to make the code
                 * consistent between GCC versions and other compilers
                 *
                 * - J.O.
                 */

#ifdef __GNUC__
#if __GNUC_PREREQ(8, 0)
                namespace fs = std::filesystem;
#else
                namespace fs = std::experimental::filesystem;
#endif
#else
                namespace fs = std::filesystem;
#endif

                fs::path nvrtc_feature_extractor_kernel_source_filepath
                    = fs::absolute(self->kernel_source_location);

                std::ifstream nvrtc_feature_extractor_kernel_source_file
                    = std::ifstream(
                        nvrtc_feature_extractor_kernel_source_filepath,
                        std::ios_base::in);

                if(!nvrtc_feature_extractor_kernel_source_file.is_open())
                {
                    throw std::runtime_error(
                        "Could not open the feature extractor kernel source "
                        "file.");
                }

                std::stringstream nvrtc_feature_extractor_kernel_source;
                nvrtc_feature_extractor_kernel_source
                    << nvrtc_feature_extractor_kernel_source_file.rdbuf();

                ptx = gst_cuda_nvrtc_compile(
                    nvrtc_feature_extractor_kernel_source.str().c_str());

                if(ptx == NULL)
                {
                    throw GstCudaException(
                        "Could not successfully compile feature extractor "
                        "kernels with NVRTC.");
                }

                if(!gst_cuda_result(
                       CuModuleLoadData(&self_private->cuda_module, ptx)))
                {
                    throw GstCudaException(
                        "Could not successfully load feature extractor "
                        "kernels with NVRTC.");
                }

                if(!gst_cuda_result(CuModuleGetFunction(
                       &self_private->feature_consolidator_kernel,
                       self_private->cuda_module,
                       GST_CUDA_FEATURE_CONSOLIDATION_KERNEL)))
                {
                    throw GstCudaException(
                        "Could not successfully load feature consolidation "
                        "kernel from NVRTC module.");
                }

                if(!gst_cuda_result(CuModuleGetFunction(
                       &(self_private->feature_extractor_kernel),
                       (self_private->cuda_module),
                       GST_CUDA_FEATURE_EXTRACTOR_KERNEL)))
                {
                    throw GstCudaException(
                        "Could not successfully load feature extractor "
                        "kernel from NVRTC module.");
                }

                if(ptx != NULL)
                {
                    g_free(ptx);
                    ptx = NULL;
                }
            }
            catch(std::exception &ex)
            {
                self_private->feature_extractor_kernel = NULL;
                self_private->feature_consolidator_kernel = NULL;

                if(self_private->cuda_module != NULL)
                {
                    CuModuleUnload(self_private->cuda_module);
                    self_private->cuda_module = NULL;
                }

                if(ptx != NULL)
                {
                    g_free(ptx);
                    ptx = NULL;
                }

                GST_ERROR_OBJECT(self, "%s", ex.what());
                result = FALSE;
            }

            gst_cuda_context_pop(NULL);
        }
        else
        {
            GST_ERROR_OBJECT(
                self,
                "Could not push CUDA context to create NVRTC CUDA module.");
            result = FALSE;
        }
    }

    return result;
}

static gboolean gst_cuda_feature_extractor_stop(GstBaseTransform *trans)
{
    GstCudaBaseTransform *filter = GST_CUDA_BASE_TRANSFORM(trans);
    GstCudaFeatureExtractor *self = GST_CUDA_FEATURE_EXTRACTOR(trans);
    GstCudaFeatureExtractorPrivate *self_private
        = gst_cuda_feature_extractor_get_instance_private_typesafe(self);

    gboolean result = TRUE;

    if(gst_cuda_context_push(filter->context))
    {
        self_private->feature_consolidator_kernel = NULL;
        self_private->feature_extractor_kernel = NULL;

        if(self_private->cuda_module != NULL)
        {
            gst_cuda_result(CuModuleUnload(self_private->cuda_module));
            self_private->cuda_module = FALSE;
        }

        gst_cuda_context_pop(NULL);
    }

    result = GST_BASE_TRANSFORM_CLASS(parent_class)->stop(trans);

    return result;
}

static GstFlowReturn gst_cuda_feature_extractor_transform_frame(
    GstCudaBaseTransform *filter,
    GstVideoFrame *in_frame,
    GstCudaMemory *in_cuda_mem,
    GstVideoFrame *out_frame,
    GstCudaMemory *out_cuda_mem)
{
    GstCudaFeatureExtractor *self = GST_CUDA_FEATURE_EXTRACTOR(filter);
    GstCudaFeatureExtractorPrivate *self_private
        = gst_cuda_feature_extractor_get_instance_private_typesafe(self);
    GstFlowReturn result = GST_FLOW_OK;

    try
    {
        GstCaps *timestamp_meta_caps
            = gst_caps_new_empty_simple("timestamp/x-utc-time");
        GstReferenceTimestampMeta *timestamp_meta
            = gst_buffer_get_reference_timestamp_meta(
                in_frame->buffer, timestamp_meta_caps);
        gst_caps_unref(timestamp_meta_caps);

        if(timestamp_meta != NULL)
        {
            self_private->frame_timestamp = timestamp_meta->timestamp;
        }
        else
        {
            self_private->frame_timestamp = GST_BUFFER_PTS(in_frame->buffer);
        }

        if(!gst_cuda_context_push(filter->context))
        {
            throw GstCudaException("Could not push CUDA context.");
        }

        GstMetaOpticalFlow *optical_flow_metadata
            = reinterpret_cast<GstMetaOpticalFlow *>(gst_buffer_get_meta(
                in_frame->buffer, gst_meta_optical_flow_api_get_type()));

        if(optical_flow_metadata != NULL)
        {
            if(self->enable_debug == TRUE)
            {
                gst_cuda_feature_extractor_output_motion_vectors(
                    self, in_frame, optical_flow_metadata);
            }

            CUDAFeaturesMatrix *features_matrix
                = gst_cuda_feature_extractor_extract_features(
                    self, optical_flow_metadata);

            GstMetaAlgorithmFeatures *algorithm_features_meta
                = GST_META_ALGORITHM_FEATURES_ADD(out_frame->buffer);

            algorithm_features_meta->features = features_matrix;

            if(self->enable_debug == TRUE)
            {
                if(algorithm_features_meta != NULL)
                {
                    gst_cuda_feature_extractor_output_features_json(
                        self, in_frame, algorithm_features_meta);
                }
            }
        }

        gst_cuda_context_pop(NULL);
    }
    catch(GstCudaException &ex)
    {
        GST_ELEMENT_ERROR(
            self,
            LIBRARY,
            FAILED,
            ("GStreamer CUDA error - %s", ex.what()),
            (NULL));

        result = GST_FLOW_ERROR;
    }
    catch(cv::Exception &ex)
    {
        GST_ELEMENT_ERROR(
            self, LIBRARY, FAILED, ("OpenCV error - %s", ex.what()), (NULL));

        result = GST_FLOW_ERROR;
    }
    catch(std::exception &ex)
    {
        GST_ELEMENT_ERROR(
            self, LIBRARY, FAILED, ("General error - %s", ex.what()), (NULL));

        result = GST_FLOW_ERROR;
    }

    self_private->frame_num++;
    return result;
}

static std::ofstream open_output_metadata_file(
    GstCudaFeatureExtractor *self,
    const GstVideoFrame *frame,
    const std::string &file_type)
{
    GstCudaFeatureExtractorPrivate *self_private
        = gst_cuda_feature_extractor_get_instance_private_typesafe(self);

    std::stringstream output_metadata_filename;

    output_metadata_filename << GST_OBJECT(self)->name << "_"
                             << frame->info.width << "x" << frame->info.height
                             << "-FR-" << std::setfill('0') << std::setw(4)
                             << self_private->frame_num + 1 << file_type;

    /*
         * Another explanation:
         *
         * Until GCC version 8, the C++17 filesystem library within the STL was
         * also included under a different namespace path
         * (std::experimental::filesystem) compared to the actual specification
         * (std::filesystem).
         *
         * As a result, we need a namespace alias in order to make the code
         * consistent between GCC versions and other compilers
         *
         * - J.O.
         */

#ifdef __GNUC__
#if __GNUC_PREREQ(8, 0)
    namespace fs = std::filesystem;
#else
    namespace fs = std::experimental::filesystem;
#endif
#else
    namespace fs = std::filesystem;
#endif

    fs::path output_metadata_filepath
        = fs::absolute(fs::current_path() / output_metadata_filename.str());

    std::ofstream output_metadata_file = std::ofstream(
        output_metadata_filepath,
        std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);

    if(!output_metadata_file.is_open())
    {
        throw std::runtime_error(
            "Could not open the optical flow motion vector output file.");
    }

    return output_metadata_file;
}

/******************************************************************************/
