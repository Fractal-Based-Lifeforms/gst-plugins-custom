// clang-format off
/**************************** Includes and Macros *****************************/
// clang-format on

#include "gstcudaof.h"

#include <stdexcept>

#include <glib-object.h>
#include <glibconfig.h>
#include <gst/base/gstbasetransform.h>
#include <gst/cuda/of/gstcudaofalgorithm.h>
#include <gst/cuda/of/gstcudaofhintvectorgridsize.h>
#include <gst/cuda/of/gstcudaofoutputvectorgridsize.h>
#include <gst/cuda/of/gstcudaofperformancepreset.h>
#include <gst/cuda/of/gstmetaopticalflow.h>
#include <gst/gst.h>
#include <gst/gstbuffer.h>
#include <gst/gstelement.h>
#include <gst/gstinfo.h>
#include <gst/gstmemory.h>
#include <gst/gstmeta.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/video/tracking.hpp>

#include <gst/cuda/nvcodec/gstcudabasetransform.h>
#include <gst/cuda/nvcodec/gstcudacontext.h>
#include <gst/cuda/nvcodec/gstcudamemory.h>
#include <gst/cuda/nvcodec/gstcudautils.h>

// clang-format off
/*
 * Just some setup for the GStreamer debug logger.
 *
 * - J.O.
 */
// clang-format on
GST_DEBUG_CATEGORY_STATIC(gst_cuda_of_debug);
#define GST_CAT_DEFAULT gst_cuda_of_debug

#define GST_CUDA_OF(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), gst_cuda_of_get_type(), GstCudaOf))

#define gst_cuda_of_parent_class parent_class

// clang-format off
/****************************** Static Variables ******************************/
// clang-format on

static const gint default_device_id = -1;
static const gboolean default_farneback_fast_pyramids = FALSE;
static const gint default_farneback_flags = 0;
static const gint default_farneback_number_of_iterations = 10;
static const gint default_farneback_number_of_levels = 10;
static const gint default_farneback_polynomial_expansion_n = 5;
static const gdouble default_farneback_polynomial_expansion_sigma = 1.1;
static const gdouble default_farneback_pyramid_scale = 0.5;
static const gint default_farneback_window_size = 13;

static const gboolean default_nvidia_enable_cost_buffer = FALSE;
static const gboolean default_nvidia_enable_external_hints = FALSE;
static const gboolean default_nvidia_enable_temporal_hints = FALSE;
static const gint default_nvidia_hint_vector_grid_size
    = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_HINT_VECTOR_GRID_SIZE_4;
static const gint default_nvidia_output_vector_grid_size
    = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
static const gint default_nvidia_performance_preset
    = cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_FAST;

static const gint default_optical_flow_algorithm
    = OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0;

// clang-format off
/**
 * \brief Anonymous enumeration containing the list of properties available for
 * the GstCudaOf GObject type this module defines.
 *
 * \notes N_PROPERTIES is a special value. N_PROPERTIES is a quick way to
 * determine the number of properties exposed by the GstCudaOf GObject type.
 */
// clang-format on
enum
{
    // clang-format off
    /**
     * \brief ID number for the GPU Device ID property.
     */
    // clang-format on
    PROP_DEVICE_ID = 1,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow fast pyramids property.
     */
    // clang-format on
    PROP_FARNEBACK_FAST_PYRAMIDS,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow flags property.
     */
    // clang-format on
    PROP_FARNEBACK_FLAGS,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow number-of-iterations
     * property.
     */
    // clang-format on
    PROP_FARNEBACK_NUMBER_OF_ITERATIONS,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow number-of-levels
     * property.
     */
    // clang-format on
    PROP_FARNEBACK_NUMBER_OF_LEVELS,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow polynomial expansion N
     * property.
     */
    // clang-format on
    PROP_FARNEBACK_POLYNOMIAL_EXPANSION_N,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow polynomial expansion
     * sigma property.
     */
    // clang-format on
    PROP_FARNEBACK_POLYNOMIAL_EXPANSION_SIGMA,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow pyramid scale property.
     */
    // clang-format on
    PROP_FARNEBACK_PYRAMID_SCALE,

    // clang-format off
    /**
     * \brief ID number for the Farneback Optical Flow window size property.
     */
    // clang-format on
    PROP_FARNEBACK_WINDOW_SIZE,

    // clang-format off
    /**
     * \brief ID number for the NVIDIA Optical Flow enable cost buffer
     * property.
     */
    // clang-format on
    PROP_NVIDIA_ENABLE_COST_BUFFER,

    // clang-format off
    /**
     * \brief ID number for the NVIDIA Optical Flow enable external hints
     * property.
     */
    // clang-format on
    PROP_NVIDIA_ENABLE_EXTERNAL_HINTS,

    // clang-format off
    /**
     * \brief ID number for the NVIDIA Optical Flow enable temporal hints
     * property.
     */
    // clang-format on
    PROP_NVIDIA_ENABLE_TEMPORAL_HINTS,

    // clang-format off
    /**
     * \brief ID number for the NVIDIA Optical Flow hint vector grid size
     * property.
     */
    // clang-format on
    PROP_NVIDIA_HINT_VECTOR_GRID_SIZE,

    // clang-format off
    /**
     * \brief ID number for the NVIDIA Optical Flow output vector grid size
     * property.
     */
    // clang-format on
    PROP_NVIDIA_OUTPUT_VECTOR_GRID_SIZE,

    // clang-format off
    /**
     * \brief ID number for the NVIDIA Optical Flow performance preset property.
     */
    // clang-format on
    PROP_NVIDIA_PERFORMANCE_PRESET,

    // clang-format off
    /**
     * \brief ID number for the Optical Flow algorithm property.
     */
    // clang-format on
    PROP_OPTICAL_FLOW_ALGORITHM,

    // clang-format off
    /**
     * \brief Number of property ID numbers in this enum.
     */
    // clang-format on
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

// clang-format off
/**
 * An array of GParamSpec instances representing the parameter specifications
 * for the parameters installed on the GstCudaOf GObject type.
 */
// clang-format on
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

// clang-format off
/**
 * \brief Sink pad template for the GstCudaOf GObject type.
 *
 * \notes The capability restrictions for this sink pad require device (GPU)
 * memory buffers for raw video data. As a result, this GStreamer element is
 * expected to be linked immediately after the nvh264dec GStreamer element or
 * another element that can support buffers with the CUDAMemory memory type.
 *
 * \notes Additionally, NV12 colour-formatting is required due to the optical
 * flow algorithms requiring single-channel grey-scale color-formatting.
 * Whilst NV12 colour-formatting is still two-channel (the Y and UV planars),
 * grey-scale colour-formatting is merely just the Y planar data
 * (luminescence). Therefore, the first channel can be used to pass grey-scale
 * information to the optical flow algorithms, as grey-scale is merely just the
 * Y planar data.
 */
// clang-format on
static GstStaticPadTemplate gst_cuda_of_sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:CUDAMemory), format = (string) NV12"));

// clang-format off
/**
 * \brief Source pad template for the GstCudaOf GObject type.
 *
 * \notes The capability restrictions for this source pad require outputting
 * device (GPU) memory buffers for raw video data. This was done because this
 * GStreamer element is not expected to modify the video data; it will simply
 * be attaching metadata to the buffer. As such, the capabilities for the
 * source pad must match the capabilities for the sink pad.
 */
// clang-format on
static GstStaticPadTemplate gst_cuda_of_src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:CUDAMemory), format = (string) NV12"));

// clang-format off
/************************** Type/Struct Definitions ***************************/
// clang-format on

// clang-format off
/*
 * \brief The structure for the GstCudaOf GStreamer element type containing the
 * public instance data.
 */
// clang-format on
typedef struct _GstCudaOf
{
    // clang-format off
    /********************************** Base **********************************/
    // clang-format on

    // clang-format off
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
    // clang-format on
    GstCudaBaseTransform parent;

    // clang-format off
    /******************************* Farneback ********************************/
    // clang-format on

    // clang-format off
    /**
     * \brief A flag that is used to enable the creation of Gaussian pyramids
     * via a CUDA kernel.
     */
    // clang-format on
    gboolean farneback_fast_pyramids;

    // clang-format off
    /**
     * \brief A series of bit-flags that are used to adjust the functionality
     * of the Farneback optical flow algorithm.
     *
     * \notes There's a couple of bit-flags available for this variable. They
     * are cv::OPTFLOW_USE_INITIAL_FLOW and cv::OPTFLOW_FARNEBACK_GUASSIAN.
     *
     * \notes There is not much documentation on these two bit-flags, but from
     * what can be gathered from the OpenCV code, the former allows the usage
     * of any vectors in the optical flow vector matrix passed to the algorithm
     * to be used as hint vectors. The latter allows for the usage of a
     * Gaussian Blur filter rather than a Box filter.
     */
    // clang-format on
    gint farneback_flags;

    // clang-format off
    /**
     * \brief The number of iterations to use for either the Gaussian Blur or
     * Box filters.
     */
    // clang-format on
    gint farneback_number_of_iterations;

    // clang-format off
    /**
     * \brief The number of levels to use for the Gaussian pyramid structures.
     */
    // clang-format on
    gint farneback_number_of_levels;

    // clang-format off
    /**
     * \brief The "N" constant value that is used by the Farneback optical flow
     * algorithm for polynomial expansion.
     *
     * \notes This appears to only accept one of two values: 5 or 7. Any other
     * value for this will result in OpenCV raising an exception.
     */
    // clang-format on
    gint farneback_polynomial_expansion_n;

    // clang-format off
    /**
     * \brief The "sigma" constant value that is used by the Farneback optical
     * flow algorithm for polynomial expansion.
     */
    // clang-format on
    gdouble farneback_polynomial_expansion_sigma;

    // clang-format off
    /**
     * \brief The scale for the Gaussian pyramid structures.
     *
     * \notes The scale is used to help determine the number of levels to use
     * for the Gaussian pyramid structures by way of cropping unnecessary
     * levels.
     *
     * \notes Also, if the fast pyramids setting is set to TRUE, then this
     * setting **MUST** be set to 0.5; otherwise OpenCV will raise an exception.
     */
    // clang-format on
    gdouble farneback_pyramid_scale;

    // clang-format off
    /**
     * \brief The window size to use for either the Gaussian Blur or Box
     * filters.
     */
    // clang-format on
    gint farneback_window_size;

    // clang-format off
    /********************************* NVIDIA *********************************/
    // clang-format on

    // clang-format off
    /**
     * \brief A flag that is used to enable the output of the cost matrix from
     * the NVIDIA optical flow algorithms.
     *
     * \notes According to the OpenCV documentation, the cost matrix documents
     * the confidence level for each vector within the output flow vector
     * matrix. The higher the cost, the lower the confidence level.
     */
    // clang-format on
    gboolean nvidia_enable_cost_buffer;

    // clang-format off
    /**
     * \brief A flag that is used to enable the input of the hint matrix into
     * the NVIDIA optical flow algorithms.
     *
     * \notes According to the OpenCV documentation, the hint matrix contains
     * flow vectors that are to be passed to the NVIDIA optical flow algorithm
     * in order to assist in calculating vectors that are considered "more
     * correct".
     */
    // clang-format on
    gboolean nvidia_enable_external_hints;

    // clang-format off
    /**
     * \brief A flag that is used to enable the re-use of the previous NVIDIA
     * optical flow calculation results as a hint matrix.
     *
     * \notes As per the documentation for nvidia_enable_external_hints, the
     * hint matrix is useful for calculating more accurate vectors. However,
     * the temporal hint matrix is derived from the previous optical flow
     * calculation. As a result, it's more useful when dealing with a video
     * stream than a series of un-related images.
     */
    // clang-format on
    gboolean nvidia_enable_temporal_hints;

    // clang-format off
    /**
     * \brief The granularity (the number of pixels each vector pair
     * represents) of the hint matrix passed to the NVIDIA optical flow 2.0
     * algorithm.
     *
     * \notes This option only works for NVIDIA optical flow 2.0 or newer and
     * only on Ampere cards or newer. Otherwise the granularity of the
     * hint matrix is limited to 4x4.
     */
    // clang-format on
    gint nvidia_hint_vector_grid_size;

    // clang-format off
    /**
     * \brief The granularity (the number of pixels each vector pair
     * represents) of the output motion vector matrix received from the NVIDIA
     * optical flow 2.0 algorithm.
     *
     * \notes This option only works for NVIDIA optical flow 2.0 or newer and
     * only on Ampere cards or newer. Otherwise the granularity of the
     * output motion vector matrix is limited to 4x4.
     */
    // clang-format on
    gint nvidia_output_vector_grid_size;

    // clang-format off
    /**
     * \brief The preset to use to determine the performance and accuracy of
     * the NVIDIA optical flow algorithms.
     *
     * \notes This option has three different settings. Of the settings
     * available, there is limited difference between the SLOW and MEDIUM
     * settings, but there is a large difference between the MEDIUM and FAST
     * settings.
     */
    // clang-format on
    gint nvidia_performance_preset;

    // clang-format off
    /******************************* Algorithms *******************************/
    // clang-format on

    // clang-format off
    /**
     * \brief The optical flow algorithm that will be used to perform optical
     * flow analysis on the received video frames.
     *
     * \notes There are three options currently implemented:
     *   - Farneback
     *   - NVIDIA Optical Flow v1.0
     *   - NVIDIA Optical Flow v2.0
     *
     * \notes The rest may be implemented at a later date.
     */
    // clang-format on
    gint optical_flow_algorithm;
} GstCudaOf;

// clang-format off
/*
 * \brief The structure for the GstCudaOf GStreamer element type containing a
 * set of pointers for the available optical flow algorithms.
 *
 * \notes This is necessary because the only class that is common to these
 * optical flow algorithms, cv::Algorithm, is a type that has no means of
 * executing the optical flow algorithms.
 *
 * \notes As a result, we need to keep separate pointers.
 */
// clang-format on
typedef struct _GstCudaOfPrivateAlgorithms
{
    // clang-format off
    /**
     * \brief A pointer to a constructed instance of a dense (one vector-pair
     * per pixel) optical flow algorithm.
     */
    // clang-format on
    cv::Ptr<cv::cuda::DenseOpticalFlow> dense_optical_flow_algorithm;

    // clang-format off
    /**
     * \brief A pointer to a constructed instance of an NVIDIA optical flow
     * algorithm.
     */
    // clang-format on
    cv::Ptr<cv::cuda::NvidiaHWOpticalFlow> nvidia_optical_flow_algorithm;

    // clang-format off
    /**
     * \brief A pointer to a constructed instance of a sparse (one vector-pair
     * per group of pixels) optical flow algorithm.
     */
    // clang-format on
    cv::Ptr<cv::cuda::SparseOpticalFlow> sparse_optical_flow_algorithm;
} GstCudaOfPrivateAlgorithms;

// clang-format off
/*
 * \brief The structure for the GstCudaOf GStreamer element type containing the
 * private instance data.
 */
// clang-format on
typedef struct _GstCudaOfPrivate
{
    // clang-format off
    /******************************** Private *********************************/
    // clang-format on

    // clang-format off
    /*
     * \brief A structure containing a set of pointers for the available
     * optical flow algorithm types.
     */
    // clang-format on
    GstCudaOfPrivateAlgorithms algorithms;

    // clang-format off
    /*
     * \brief A flag to determine if the algorithm has been initialised or not.
     *
     * \notes This is to avoid re-initialising the algorithm each and every
     * time we need to calculate the optical flow data. The algorithm is only
     * initialised in the PLAYING state once at least two buffers have been
     * received.
     *
     * \notes If the element has been reset to NULL, then the algorithm will
     * need to be reinitialised once again.
     */
    // clang-format on
    gboolean algorithm_is_initialised;

    // clang-format off
    /**
     * \brief A pointer to a copy of the previously received buffer.
     *
     * \notes We need this due to the optical flow algorithms requiring two
     * frames. In particular, this is a pointer to a copy, rather than a second
     * reference to the buffer. This is to avoid the buffer becoming
     * un-writable later in the pipeline; as buffers are only writable when a
     * single reference to them is being held.
     *
     * \notes As a buffer holds a reference to the memory, and copying
     * increases the reference count to this memory, copying the buffer does
     * not require the memory itself to be copied. As a result, it is a fairly
     * quick, but necessary process.
     */
    // clang-format on
    GstBuffer *prev_buffer;
} GstCudaOfPrivate;

// clang-format off
/**
 * \brief The structure for the GstCudaOf GStreamer element type containing the
 * class' data.
 *
 * \details This structure contains class-level data and the virtual function
 * table for itself and its parent classes. This structure will be shared
 * between all instances of the GstCudaOf type.
 */
// clang-format on
typedef struct _GstCudaOfClass
{
    // clang-format off
    /********************************** Base **********************************/
    // clang-format on

    // clang-format off
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
    // clang-format on
    GstCudaBaseTransformClass parent_class;
} GstCudaOfClass;

// clang-format off
/**
 * \brief Exception representing errors utilising GStreamer's CUDA
 * methods/data-types.
 *
 * \details This exception represents serveral potential errors that can occur
 * during the usage of the GstCudaContext data-type, or other possible GStreamer
 * CUDA methods/data-types.
 */
// clang-format on
class GstCudaException : public std::logic_error
{
    public:
    // clang-format off
    /************************ Public Member Functions *************************/
    // clang-format on

    // clang-format off
    /**
     * \brief Default constructor.
     *
     * \details Constructs an instance of this exception with the given
     * message.
     */
    // clang-format on
    explicit GstCudaException(const std::string &what_arg)
        : std::logic_error(what_arg)
    {
    }

    // clang-format off
    /**
     * \brief Default constructor.
     *
     * \details Constructs an instance of this exception with the given
     * message.
     */
    // clang-format on
    explicit GstCudaException(const char *what_arg) : std::logic_error(what_arg)
    {
    }

    // clang-format off
    /**
     * \brief Copy constructor.
     *
     * \details Constructs an instance of this exception with the message from
     * another instance of this exception.
     */
    // clang-format on
    GstCudaException(const GstCudaException &other) noexcept
        : GstCudaException(other.what())
    {
    }

    // clang-format off
    /**
     * \brief Returns the instance's message.
     *
     * \returns The instance's message.
     */
    // clang-format on
    const char *what() const noexcept override
    {
        return std::logic_error::what();
    }
};

// clang-format off
/*************************** Function Declarations ****************************/
// clang-format on

// clang-format off
/**
 * \brief Calculates optical flow between the two given buffers and returns the
 * result as a 2-channel 2D matrix hosted on the GPU.
 *
 * \details This function extracts the pointers to the memory within the GPU
 * for the two buffers. It then uses these pointers to construct GPU matrix
 * instances as wrappers and uses the appropriate optical flow algorithm to
 * calculate and output the optical flow vectors.
 *
 * \details The optical flow vectors are then returned as a cv::cuda::GpuMat
 * instance, representing a 2-channel 2D matrix hosted on the GPU.
 *
 * \param[in] self An instance of the GstCudaOf GObject type.
 * \param[in] current_buffer The current buffer being processed by the element
 * to calculate optical flow vectors for.
 * \param[in] previous_buffer The previous buffer that has been processed by
 * the element to use for the calculation of optical flow vectors.
 *
 * \exception cv::Exception If an error occurred during the usage of one of the
 * OpenCV optical flow algorithms
 *
 * \returns A cv::cuda::GpuMat instance, representing a 2-channel 2D matrix
 * hosted on the GPU.
 *
 */
// clang-format on
static cv::cuda::GpuMat gst_cuda_of_calculate_optical_flow(
    GstCudaOf *self,
    GstBuffer *current_buffer,
    GstBuffer *previous_buffer);

// clang-format off
/**
 * \brief Extracts the GstCudaMemory pointer from a buffer.
 *
 * \details Under the assumption that the given buffer has only a single
 * GstCudaMemory memory instance, and the memory allocated by that
 * GstCudaMemory instance is acessible from the current context, the function
 * will attempt to extract and return the pointer to that GstCudaMemory
 * instance. Otherwise, it will return NULL.
 *
 * \param[in] self An instance of the GstCudaOf GObject type.
 * \param[in] buf An instance of a GstBuffer to extract the GstCudaMemory
 * instance from.
 *
 * \returns A pointer to the GstCudaMemory instance held by the given GstBuffer
 * instance. Alternatively, NULL is returned if a GstCudaMemory could not be
 * found or the device memory pointer is not accessble from the current CUDA
 * context.
 */
// clang-format on
static GstCudaMemory *
gst_cuda_of_get_cuda_memory(GstCudaOf *self, GstBuffer *buf);

// clang-format off
/**
 * \brief Wrapper around gst_cuda_of_get_instance_private.
 *
 * \details Wraps around gst_cuda_of_get_instance_private in order to return a
 * pointer to a GstCudaOfPrivate instance.
 *
 * \param[in] self A GstCudaOf GObject instance to get the property from.
 *
 * \returns A pointer to the GstCudaOfPrivate instance for the given GstCudaOf
 * instance.
 */
// clang-format on
static GstCudaOfPrivate *
gst_cuda_of_get_instance_private_typesafe(GstCudaOf *self);

// clang-format off
/**
 * \brief Property getter for instances of the GstCudaOf GObject.
 *
 * \details Gets properties from an instance of the GstCudaOf GObject.
 *
 * \details Any property ID that does not exist on the GstCudaOfClass
 * GClassObject will result in that property being ignored by the getter.
 *
 * \param[in] gobject A GstCudaOf GObject instance to get the property from.
 * \param[in] prop_id The ID number for the property to get. This should
 * correspond to an entry in the enum defined in this file.
 * \param[out] value The GValue instance (generic value container) that will
 * contain the value of the property after being converted into the generic
 * GValue type.
 * \param[in] pspec The property specification instance for the property that
 * we are getting. This property specification should exist on the
 * GstCudaOfClass GObjectClass structure.
 */
// clang-format on
static void gst_cuda_of_get_property(
    GObject *gobject,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec);

// clang-format off
/**
 * \brief Initialisation function for the optical flow algorithms.
 *
 * \details This is a function that is used to initialise an OpenCV optical
 * flow algorithm on a GstCudaOf instance. This function is called the first
 * time an optical flow algorithm is required for the calculation of optical
 * flow vectors.
 *
 * \details Depending on the type of algorithm required, the OpenCV algorithm
 * type will be initialised with their static creation function and will be
 * assigned to one of the three available optical flow algorithm pointer types
 * (dense, NVIDIA, sparse).
 *
 * \param[in,out] self An instance of the GstCudaOf GObject type.
 * \param[in] algorithm_type An enumeration value representing the type of
 * OpenCV optical flow algorithm to initialise and make available to the
 * element.
 *
 * \exception cv::Exception If an error occurred during the
 * construction/initialisation of one of the OpenCV optical flow algorithms
 *
 * \notes Due to the way the element is setup, only one algorithm can be
 * initialised at a time. So this function should only be called once per
 * transition to the PLAYING state for the element.
 */
// clang-format on
static void
gst_cuda_of_init_algorithm(GstCudaOf *self, GstCudaOfAlgorithm algorithm_type);

// clang-format off
/**
 * \brief Property setter for instances of the GstCudaOf GObject.
 *
 * \details Sets properties on an instance of the GstCudaOf GObject.
 *
 * \details Any property ID that does not exist on the GstCudaOfClass
 * GClassObject will result in that property being ignored by the setter.
 *
 * \param[in,out] gobject A GstCudaOf GObject instance to set the property on.
 * \param[in] prop_id The ID number for the property to set. This should
 * correspond to an entry in the enum defined in this file.
 * \param[in] value The GValue instance (generic value container) that will be
 * convered to the correct value type to set the property with.
 * \param[in] pspec The property specification instance for the property that
 * we are setting. This property specification should exist on the GstCudaOfClass
 * GObjectClass structure.
 */
// clang-format on
static void gst_cuda_of_set_property(
    GObject *gobject,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec);

// clang-format off
/**
 * \brief Sets up the element to begin processing.
 *
 * \details This method sets up the element for processing by calling the
 * parent class' implementation of this method. Then the flag for algorithm
 * initialisation is cleared and the pointer to the previous buffer is also
 * cleared ready for incoming data to be processed.
 *
 * \param[in,out] trans A GstCudaOf GObject instance to setup for processing.
 *
 * \returns TRUE if setup was successful. False if errors occurred.
 */
// clang-format on
static gboolean gst_cuda_of_start(GstBaseTransform *trans);

// clang-format off
/**
 * \brief Cleans up the element to stop processing.
 *
 * \details This method cleans up the element by clearing the pointer to the
 * previous buffer and the flag for algorithm initialisation. The pointers to
 * the OpenCV optical flow algorithms are also cleared to free up resources
 * allocated to them. Finally, the parent class' implementation of the method
 * is called.
 *
 * \details After this method is called, no more processing is expected to be
 * performed by the element until it has been transitioned back into the
 * PLAYING state.
 *
 * \param[in,out] trans A GstCudaOf GObject instance to clean-up and stop
 * processing on.
 *
 * \returns TRUE if clean-up was successful. False if errors occurred.
 */
// clang-format on
static gboolean gst_cuda_of_stop(GstBaseTransform *trans);

// clang-format off
/**
 * \brief Calculates the optical flow between the previous and current buffers,
 * and attaches it as metadata to the output buffer.
 *
 * \details This method uses OpenCV to calculate optical flow vectors between
 * the current input buffer and the previous input buffer. The result of this
 * calculation is then attached as metadata to the pointer to the current
 * output buffer and passed to the next attached element.
 *
 * \param[in] trans A GstCudaOf GObject instance.
 * \param[in] inbuf A pointer to the buffer that the GstCudaOf element has
 * received on its sink pad.
 * \param[in,out] outbuf A pointer to the buffer that the GstCudaOf element
 * will output on its source pad.
 *
 * \returns GST_FLOW_OK if no errors occurred. GST_FLOW_ERROR if an error
 * occurred during the initialisation of the optical flow algorithm and/or
 * calculation of the optical flow vectors.
 *
 * \notes This method is used over the `transform_ip` method due to the
 * requirement of attaching metadata to the buffer. Since this requires a
 * writable buffer, it will not work if we're dealing with buffers with more
 * than a single reference; which is possible if the `tee` element is used
 * within a pipeline.
 *
 * \notes Therefore, the `transform` method is used instead, and a copy of the
 * input buffer is returned from the element with the attached optical flow
 * vectors as metadata.
 */
// clang-format on
static GstFlowReturn gst_cuda_of_transform(
    GstBaseTransform *trans,
    GstBuffer *inbuf,
    GstBuffer *outbuf);

// clang-format off
/************************** GObject Type Definitions **************************/
// clang-format on

/*
 * Okay, so this is the main GObject meta-programming macro for defining the
 * type. This macro generates a bunch of things; including the
 * gst_cuda_of_get_type function, which defines the actual GObject
 * type during runtime.
 *
 * - J.O.
 */
G_DEFINE_TYPE_WITH_PRIVATE(GstCudaOf, gst_cuda_of, GST_TYPE_CUDA_BASE_TRANSFORM)

// clang-format off
/**************************** Function Definitions ****************************/
// clang-format on

static cv::cuda::GpuMat gst_cuda_of_calculate_optical_flow(
    GstCudaOf *self,
    GstBuffer *current_buffer,
    GstBuffer *previous_buffer)
{

    GstCudaMemory *current_buffer_cuda_memory = NULL;
    GstCudaMemory *prev_buffer_cuda_memory = NULL;

    gboolean current_buffer_map_result = FALSE;
    gboolean prev_buffer_map_result = FALSE;

    GstMapInfo current_buffer_map_info;
    GstMapInfo prev_buffer_map_info;

    GstCudaOfPrivate *self_private
        = gst_cuda_of_get_instance_private_typesafe(self);

    cv::cuda::GpuMat optical_flow_gpu_mat = cv::cuda::GpuMat(
        self->parent.in_info.height, self->parent.in_info.width, CV_32FC2);

    current_buffer_cuda_memory
        = gst_cuda_of_get_cuda_memory(self, current_buffer);
    prev_buffer_cuda_memory
        = gst_cuda_of_get_cuda_memory(self, previous_buffer);

    if(current_buffer_cuda_memory != NULL && prev_buffer_cuda_memory != NULL)
    {
        current_buffer_map_result = gst_memory_map(
            GST_MEMORY_CAST(current_buffer_cuda_memory),
            &current_buffer_map_info,
            (GstMapFlags)(GST_MAP_CUDA));
        prev_buffer_map_result = gst_memory_map(
            GST_MEMORY_CAST(prev_buffer_cuda_memory),
            &prev_buffer_map_info,
            (GstMapFlags)(GST_MAP_CUDA));

        if(current_buffer_map_result && prev_buffer_map_result)
        {
            cv::cuda::GpuMat current_buffer_gpu_mat = cv::cuda::GpuMat(
                self->parent.in_info.height,
                self->parent.in_info.width,
                CV_8UC1,
                current_buffer_map_info.data,
                current_buffer_cuda_memory->stride);
            cv::cuda::GpuMat prev_buffer_gpu_mat = cv::cuda::GpuMat(
                self->parent.in_info.height,
                self->parent.in_info.width,
                CV_8UC1,
                prev_buffer_map_info.data,
                prev_buffer_cuda_memory->stride);

            switch(self->optical_flow_algorithm)
            {
                case OPTICAL_FLOW_ALGORITHM_FARNEBACK:
                    {
                        self_private->algorithms.dense_optical_flow_algorithm
                            ->calc(
                                prev_buffer_gpu_mat,
                                current_buffer_gpu_mat,
                                optical_flow_gpu_mat);
                    }
                    break;
                case OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0:
                    {
                        cv::cuda::GpuMat downsampled_optical_flow_gpu_mat;
                        self_private->algorithms.nvidia_optical_flow_algorithm
                            ->calc(
                                prev_buffer_gpu_mat,
                                current_buffer_gpu_mat,
                                downsampled_optical_flow_gpu_mat);
                        std::dynamic_pointer_cast<
                            cv::cuda::NvidiaOpticalFlow_1_0>(
                            self_private->algorithms
                                .nvidia_optical_flow_algorithm)
                            ->upSampler(
                                downsampled_optical_flow_gpu_mat,
                                cv::Size(
                                    self->parent.in_info.width,
                                    self->parent.in_info.height),
                                self_private->algorithms
                                    .nvidia_optical_flow_algorithm
                                    ->getGridSize(),
                                optical_flow_gpu_mat);
                    }
                    break;
                case OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0:
                    {
                        cv::cuda::GpuMat downsampled_optical_flow_gpu_mat;
                        self_private->algorithms.nvidia_optical_flow_algorithm
                            ->calc(
                                prev_buffer_gpu_mat,
                                current_buffer_gpu_mat,
                                downsampled_optical_flow_gpu_mat);
                        std::dynamic_pointer_cast<
                            cv::cuda::NvidiaOpticalFlow_2_0>(
                            self_private->algorithms
                                .nvidia_optical_flow_algorithm)
                            ->convertToFloat(
                                downsampled_optical_flow_gpu_mat,
                                optical_flow_gpu_mat);
                    }
                    break;
                default:
                    break;
            }
        }

        if(current_buffer_map_result)
        {
            gst_memory_unmap(
                GST_MEMORY_CAST(current_buffer_cuda_memory),
                &current_buffer_map_info);
        }

        if(prev_buffer_map_result)
        {
            gst_memory_unmap(
                GST_MEMORY_CAST(prev_buffer_cuda_memory),
                &prev_buffer_map_info);
        }
    }

    return optical_flow_gpu_mat;
}

// clang-format off
/**
 * \brief Initialisation function for the GstCudaOfClass GObjectClass type.
 *
 * \details This is a class initialisation function that is called the first
 * time a GstCudaOf instance is allocated.
 *
 * \details The class structure is setup with the necessary properties, virtual
 * method overrides, element pads and element metadata for a GObject class
 * derived from one of GStreamer's base element types.
 *
 * \param[in,out] klass The instance of the GstCudaOfClass GObjectClass
 * structure.
 */
// clang-format on
static void gst_cuda_of_class_init(GstCudaOfClass *klass)
{
    /*
     * Okay, so this is one of the main two initialisation functions. This will
     * get called the first time an instance of the GstCudaOf GObject
     * type is created (typically via g_object_new). This will let us setup the
     * GstCudaOfClass GObjectClass structure for all
     * GstCudaOf instances.
     *
     * This allows us to setup properties, virtual function pointers (though
     * only our base classes have them for now) and potentially even class-level
     * variables.
     *
     * - J.O.
     */
    GObjectClass *gobject_class = NULL;
    GstElementClass *gstelement_class = NULL;
    GstBaseTransformClass *gstbasetransform_class = NULL;

    gobject_class = G_OBJECT_CLASS(klass);
    gstelement_class = GST_ELEMENT_CLASS(klass);
    gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_cuda_of_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_cuda_of_get_property);

    properties[PROP_DEVICE_ID] = g_param_spec_int(
        "cuda-device-id",
        "Cuda Device ID",
        "Set the GPU device to use for operations (-1 = auto)",
        -1,
        G_MAXINT,
        default_device_id,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_FAST_PYRAMIDS] = g_param_spec_boolean(
        "farneback-fast-pyramids",
        "Farneback Enable Fast Pyramids",
        "Enables the creation of the Gaussian pyramid structures via a CUDA "
        "kernel.",
        default_farneback_fast_pyramids,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_FLAGS] = g_param_spec_int(
        "farneback-flags",
        "Farneback Option Flags",
        "Sets a bundle of option flags to adjust the functionality of the "
        "Farneback optical flow algorithm. These include "
        "cv::OPTFLOW_USE_INITIAL_FLOW (uses the flow vectors given as hints) "
        "and cv::OPTFLOW_FARNEBACK_GAUSSIAN (uses a Gaussian Blur filter "
        "instead of a Box filter).",
        0,
        (int)(cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_FARNEBACK_GAUSSIAN),
        default_farneback_flags,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_NUMBER_OF_ITERATIONS] = g_param_spec_int(
        "farneback-number-of-iterations",
        "Farneback Number Of Iterations",
        "Sets the number of iterations to use for the Gaussian Blur or Box "
        "filters.",
        0,
        G_MAXINT,
        default_farneback_number_of_iterations,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_NUMBER_OF_LEVELS] = g_param_spec_int(
        "farneback-number-of-levels",
        "Farneback Number Of Levels",
        "Sets the number of levels to use for the Gaussian pyramid structures.",
        0,
        G_MAXINT,
        default_farneback_number_of_levels,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_POLYNOMIAL_EXPANSION_N] = g_param_spec_int(
        "farneback-polynomial-expansion-n",
        "Farneback Polynomial Expansion N",
        "Sets the N constant value that is used in polynomial expansion (can "
        "only be set to 5 or 7).",
        5,
        7,
        default_farneback_polynomial_expansion_n,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_POLYNOMIAL_EXPANSION_SIGMA] = g_param_spec_double(
        "farneback-polynomial-expansion-sigma",
        "Farneback Polynomial Expansion Sigma",
        "Sets the sigma constant value that is used in polynomial expansion.",
        0.0f,
        G_MAXDOUBLE,
        default_farneback_polynomial_expansion_sigma,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_PYRAMID_SCALE] = g_param_spec_double(
        "farneback-pyramid-scale",
        "Farneback Pyramid Scale",
        "Sets the scale for the pyramid, which is used to determine the number "
        "of levels used for the pyramid. If using the fast pyramids setting, "
        "the value for the pyramid scale must be 0.5.",
        0.0f,
        G_MAXDOUBLE,
        default_farneback_pyramid_scale,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FARNEBACK_WINDOW_SIZE] = g_param_spec_int(
        "farneback-window-size",
        "Farneback Window Size",
        "Sets the size of the window that is used for the Gaussian Blur or Box "
        "filters .",
        0,
        G_MAXINT,
        default_farneback_window_size,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_NVIDIA_ENABLE_COST_BUFFER] = g_param_spec_boolean(
        "nvidia-enable-cost-buffer",
        "NVIDIA Enable Cost Buffer",
        "Enables the output of the cost buffer from the NVIDIA Optical Flow "
        "hardware algorithms.",
        default_nvidia_enable_cost_buffer,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_NVIDIA_ENABLE_EXTERNAL_HINTS] = g_param_spec_boolean(
        "nvidia-enable-external-hints",
        "NVIDIA Enable External Hints",
        "Enables the usage of an optional external hints buffer that can be "
        "passed to the NVIDIA Optical Flow hardware algorithms.",
        default_nvidia_enable_external_hints,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_NVIDIA_ENABLE_TEMPORAL_HINTS] = g_param_spec_boolean(
        "nvidia-enable-temporal-hints",
        "NVIDIA Enable Temporal Hints",
        "Enables the usage of an internal temporal hints buffer that is "
        "stored between optical flow calculations by the NVIDIA Optical Flow "
        "hardware algorithms. The temporal hints buffer is useful when "
        "performing optical flow on several consecutive video frames.",
        default_nvidia_enable_temporal_hints,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_NVIDIA_HINT_VECTOR_GRID_SIZE] = g_param_spec_enum(
        "nvidia-hint-vector-grid-size",
        "NVIDIA Hint Vector Grid Size",
        "Sets the grid size of the hint vectors that are passed to the NVIDIA "
        "Optical Flow hardware algorithms.",
        gst_cuda_of_hint_vector_grid_size_get_type(),
        default_nvidia_hint_vector_grid_size,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_NVIDIA_OUTPUT_VECTOR_GRID_SIZE] = g_param_spec_enum(
        "nvidia-output-vector-grid-size",
        "NVIDIA Output Vector Grid Size",
        "Sets the grid size of the output vectors that are received from the "
        "NVIDIA Optical Flow hardware algorithms.",
        gst_cuda_of_output_vector_grid_size_get_type(),
        default_nvidia_output_vector_grid_size,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    /*
     * As a general note, the performance gains and quality losses that come
     * from selecting the MEDIUM preset vs the SLOW preset are fairly minimal.
     * The quality losses and performance gains from picking the FAST preset,
     * however, are much more drastic.
     *
     * I would recommend choosing between either the SLOW or FAST preset. The
     * MEDIUM preset is too close to the SLOW preset to really be of much use.
     *
     * - J.O.
     */
    properties[PROP_NVIDIA_PERFORMANCE_PRESET] = g_param_spec_enum(
        "nvidia-performance-preset",
        "NVIDIA Optical Flow Performance Preset",
        "Sets the performance preset for the NVIDIA Optical Flow hardware "
        "algorithms. The performance presets range from slow ("
        "highest-quality, but slowest performance), medium (median quality "
        "and performance) and fast (lowest-quality, but fastest performance).",
        gst_cuda_of_performance_preset_get_type(),
        default_nvidia_performance_preset,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    properties[PROP_OPTICAL_FLOW_ALGORITHM] = g_param_spec_enum(
        "optical-flow-algorithm",
        "CUDA Optical Flow Algorithm",
        "Chooses the available CUDA (or hardware) optical flow algorithm to "
        "use to perform the optical flow analysis on the incoming buffers.",
        gst_cuda_of_algorithm_get_type(),
        default_optical_flow_algorithm,
        (GParamFlags)(G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY | G_PARAM_STATIC_STRINGS));

    g_object_class_install_properties(gobject_class, N_PROPERTIES, properties);

    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_cuda_of_sink_template));
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_cuda_of_src_template));

    gst_element_class_set_metadata(
        gstelement_class,
        "CUDA Optical flow",
        "Filter/Video/Hardware",
        "Wrapper around OpenCV's optical flow implementations to extract "
        "optical flow data and store it as buffer metadata.",
        "icetana");

    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_cuda_of_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_cuda_of_stop);
    gstbasetransform_class->transform
        = GST_DEBUG_FUNCPTR(gst_cuda_of_transform);

    gstbasetransform_class->passthrough_on_same_caps = FALSE;
    gstbasetransform_class->transform_ip_on_passthrough = FALSE;
}

static GstCudaMemory *
gst_cuda_of_get_cuda_memory(GstCudaOf *self, GstBuffer *buf)
{
    GstMemory *buffer_memory = NULL;
    GstCudaMemory *buffer_cuda_memory = NULL;
    GstCudaMemory *result = NULL;

    if(gst_buffer_n_memory(buf) == 1)
    {
        buffer_memory = gst_buffer_peek_memory(buf, 0);
        if(gst_is_cuda_memory(buffer_memory))
        {
            buffer_cuda_memory = GST_CUDA_MEMORY_CAST(buffer_memory);

            if(buffer_cuda_memory->context == self->parent.context)
            {
                result = buffer_cuda_memory;
            }
            else if(
                gst_cuda_context_get_handle(buffer_cuda_memory->context)
                == gst_cuda_context_get_handle(self->parent.context))
            {
                result = buffer_cuda_memory;
            }
            else if(
                gst_cuda_context_can_access_peer(
                    buffer_cuda_memory->context, self->parent.context)
                && gst_cuda_context_can_access_peer(
                    self->parent.context, buffer_cuda_memory->context))
            {
                result = buffer_cuda_memory;
            }
        }
    }

    return result;
}

static GstCudaOfPrivate *
gst_cuda_of_get_instance_private_typesafe(GstCudaOf *self)
{
    return static_cast<GstCudaOfPrivate *>(
        gst_cuda_of_get_instance_private(self));
};

static void gst_cuda_of_get_property(
    GObject *gobject,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec)
{
    GstCudaOf *gst_cuda_of = GST_CUDA_OF(gobject);

    g_assert_cmpint(prop_id, !=, 0);
    g_assert_cmpint(prop_id, !=, N_PROPERTIES);
    g_assert(pspec == properties[prop_id]);

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
            g_value_set_int(value, gst_cuda_of->parent.device_id);
            break;
        case PROP_FARNEBACK_FAST_PYRAMIDS:
            g_value_set_boolean(value, gst_cuda_of->farneback_fast_pyramids);
            break;
        case PROP_FARNEBACK_FLAGS:
            g_value_set_int(value, gst_cuda_of->farneback_flags);
            break;
        case PROP_FARNEBACK_NUMBER_OF_ITERATIONS:
            g_value_set_int(value, gst_cuda_of->farneback_number_of_iterations);
            break;
        case PROP_FARNEBACK_NUMBER_OF_LEVELS:
            g_value_set_int(value, gst_cuda_of->farneback_number_of_levels);
            break;
        case PROP_FARNEBACK_POLYNOMIAL_EXPANSION_N:
            g_value_set_int(
                value, gst_cuda_of->farneback_polynomial_expansion_n);
            break;
        case PROP_FARNEBACK_POLYNOMIAL_EXPANSION_SIGMA:
            g_value_set_double(
                value, gst_cuda_of->farneback_polynomial_expansion_sigma);
            break;
        case PROP_FARNEBACK_PYRAMID_SCALE:
            g_value_set_double(value, gst_cuda_of->farneback_pyramid_scale);
            break;
        case PROP_FARNEBACK_WINDOW_SIZE:
            g_value_set_int(value, gst_cuda_of->farneback_window_size);
            break;
        case PROP_NVIDIA_ENABLE_COST_BUFFER:
            g_value_set_boolean(value, gst_cuda_of->nvidia_enable_cost_buffer);
            break;
        case PROP_NVIDIA_ENABLE_EXTERNAL_HINTS:
            g_value_set_boolean(
                value, gst_cuda_of->nvidia_enable_external_hints);
            break;
        case PROP_NVIDIA_ENABLE_TEMPORAL_HINTS:
            g_value_set_boolean(
                value, gst_cuda_of->nvidia_enable_temporal_hints);
            break;
        case PROP_NVIDIA_HINT_VECTOR_GRID_SIZE:
            g_value_set_enum(value, gst_cuda_of->nvidia_hint_vector_grid_size);
            break;
        case PROP_NVIDIA_OUTPUT_VECTOR_GRID_SIZE:
            g_value_set_enum(
                value, gst_cuda_of->nvidia_output_vector_grid_size);
            break;
        case PROP_NVIDIA_PERFORMANCE_PRESET:
            g_value_set_enum(value, gst_cuda_of->nvidia_performance_preset);
            break;
        case PROP_OPTICAL_FLOW_ALGORITHM:
            g_value_set_enum(value, gst_cuda_of->optical_flow_algorithm);
            break;
        default:
            g_assert_not_reached();
    }
}

// clang-format off
/**
 * \brief Initialisation function for GstCudaOf instances.
 *
 * \details This is an instance initialisation function that is called each
 * time a GstCudaOf instance is allocated.
 *
 * \details The object structure for the GstCudaOf instance is setup with the
 * necessary property defaults.
 *
 * \param[in,out] self An instance of the GstCudaOf GObject type.
 */
// clang-format on
static void gst_cuda_of_init(GstCudaOf *self)
{
    /*
     * This is the other of the two main initialisation functions for the
     * GstCudaOf GObject type. This one is called each and every time
     * a new instance of the type is created. It sets up the flags for the base
     * class and the default values for the properties.
     *
     * - J.O.
     */
    GstBaseTransform *trans = GST_BASE_TRANSFORM(self);
    GstCudaOfPrivate *self_private
        = gst_cuda_of_get_instance_private_typesafe(self);

    self->parent.device_id = default_device_id;

    self->farneback_fast_pyramids = default_farneback_fast_pyramids;
    self->farneback_flags = default_farneback_flags;
    self->farneback_number_of_iterations
        = default_farneback_number_of_iterations;
    self->farneback_number_of_levels = default_farneback_number_of_levels;
    self->farneback_polynomial_expansion_n
        = default_farneback_polynomial_expansion_n;
    self->farneback_polynomial_expansion_sigma
        = default_farneback_polynomial_expansion_sigma;
    self->farneback_pyramid_scale = default_farneback_pyramid_scale;
    self->farneback_window_size = default_farneback_window_size;

    self->nvidia_enable_cost_buffer = default_nvidia_enable_cost_buffer;
    self->nvidia_enable_external_hints = default_nvidia_enable_external_hints;
    self->nvidia_enable_temporal_hints = default_nvidia_enable_temporal_hints;
    self->nvidia_hint_vector_grid_size = default_nvidia_hint_vector_grid_size;
    self->nvidia_output_vector_grid_size
        = default_nvidia_output_vector_grid_size;
    self->nvidia_performance_preset = default_nvidia_performance_preset;

    self->optical_flow_algorithm = default_optical_flow_algorithm;

    self_private->algorithm_is_initialised = FALSE;
    self_private->prev_buffer = NULL;

    gst_base_transform_set_in_place(trans, TRUE);
    gst_base_transform_set_gap_aware(trans, FALSE);
    gst_base_transform_set_passthrough(trans, FALSE);
    gst_base_transform_set_prefer_passthrough(trans, FALSE);
}

static void
gst_cuda_of_init_algorithm(GstCudaOf *self, GstCudaOfAlgorithm algorithm_type)
{
    GstCudaOfPrivate *self_private
        = gst_cuda_of_get_instance_private_typesafe(self);
    gint device_id = self->parent.device_id;

    g_object_get(self->parent.context, "cuda-device-id", &device_id, NULL);

    switch(algorithm_type)
    {
        case OPTICAL_FLOW_ALGORITHM_FARNEBACK:
            self_private->algorithms.dense_optical_flow_algorithm
                = cv::cuda::FarnebackOpticalFlow::create(
                    self->farneback_number_of_levels,
                    self->farneback_pyramid_scale,
                    self->farneback_fast_pyramids,
                    self->farneback_window_size,
                    self->farneback_number_of_iterations,
                    self->farneback_polynomial_expansion_n,
                    self->farneback_polynomial_expansion_sigma,
                    self->farneback_flags);
            self_private->algorithm_is_initialised = TRUE;
            break;
        case OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0:
            self_private->algorithms.nvidia_optical_flow_algorithm
                = cv::cuda::NvidiaOpticalFlow_1_0::create(
                    cv::Size(
                        self->parent.in_info.width,
                        self->parent.in_info.height),
                    (cv::cuda::NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL)(
                        self->nvidia_performance_preset),
                    self->nvidia_enable_temporal_hints,
                    self->nvidia_enable_external_hints,
                    self->nvidia_enable_cost_buffer,
                    device_id);
            self_private->algorithm_is_initialised = TRUE;
            break;
        case OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0:
            self_private->algorithms.nvidia_optical_flow_algorithm
                = cv::cuda::NvidiaOpticalFlow_2_0::create(
                    cv::Size(
                        self->parent.in_info.width,
                        self->parent.in_info.height),
                    (cv::cuda::NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL)(
                        self->nvidia_performance_preset),
                    (cv::cuda::NvidiaOpticalFlow_2_0::
                         NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE)(
                        self->nvidia_output_vector_grid_size),
                    (cv::cuda::NvidiaOpticalFlow_2_0::
                         NVIDIA_OF_HINT_VECTOR_GRID_SIZE)(
                        self->nvidia_hint_vector_grid_size),
                    self->nvidia_enable_temporal_hints,
                    self->nvidia_enable_external_hints,
                    self->nvidia_enable_cost_buffer,
                    device_id);
            self_private->algorithm_is_initialised = TRUE;
            break;
        default:
            break;
    }
}

gboolean gst_cuda_of_plugin_init(GstPlugin *plugin)
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
        gst_cuda_of_debug, "cudaof", 0, "CUDA Optical flow");

    return gst_element_register(
        plugin, "cudaof", GST_RANK_NONE, gst_cuda_of_get_type());
}

static void gst_cuda_of_set_property(
    GObject *gobject,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec)
{
    GstCudaOf *gst_cuda_of = GST_CUDA_OF(gobject);

    /*
     * Just a few assertions that I noticed in the test code for GObject that
     * I'll be keeping here. If the assertions fail, then errors will be
     * printed to the console. Useful for debugging and for diagnosing unforseen
     * issues in the field; so I'll be leaving them in for now.
     *
     * - J.O.
     */
    g_assert_cmpint(prop_id, !=, 0);
    g_assert_cmpint(prop_id, !=, N_PROPERTIES);
    g_assert(pspec == properties[prop_id]);

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
            if(gst_cuda_of->parent.device_id != g_value_get_int(value))
            {
                gst_cuda_of->parent.device_id = g_value_get_int(value);
                g_assert(properties[PROP_DEVICE_ID] != NULL);
                g_object_notify_by_pspec(gobject, properties[PROP_DEVICE_ID]);
            }
            break;
        case PROP_FARNEBACK_FAST_PYRAMIDS:
            if(gst_cuda_of->farneback_fast_pyramids
               != g_value_get_boolean(value))
            {
                if(g_value_get_boolean(value) == FALSE)
                {
                    gst_cuda_of->farneback_fast_pyramids
                        = g_value_get_boolean(value);
                    g_assert(properties[PROP_FARNEBACK_FAST_PYRAMIDS] != NULL);
                    g_object_notify_by_pspec(
                        gobject, properties[PROP_FARNEBACK_FAST_PYRAMIDS]);
                }
                else if(
                    g_value_get_boolean(value) == TRUE
                    && G_APPROX_VALUE(
                           gst_cuda_of->farneback_pyramid_scale, 0.5, 1e-6)
                           == TRUE)
                {
                    gst_cuda_of->farneback_fast_pyramids
                        = g_value_get_boolean(value);
                    g_assert(properties[PROP_FARNEBACK_FAST_PYRAMIDS] != NULL);
                    g_object_notify_by_pspec(
                        gobject, properties[PROP_FARNEBACK_FAST_PYRAMIDS]);
                }
                else
                {
                    GST_WARNING_OBJECT(
                        gst_cuda_of,
                        "Could not set the farneback-fast-pyramids property. "
                        "The farneback-pyramid-scale property was not set to "
                        "0.5 when the given value is TRUE. Leaving at the "
                        "previous value.");
                }
            }
            break;
        case PROP_FARNEBACK_FLAGS:
            if(gst_cuda_of->farneback_flags != g_value_get_int(value))
            {
                if((g_value_get_int(value)
                    & (cv::OPTFLOW_FARNEBACK_GAUSSIAN
                       | cv::OPTFLOW_USE_INITIAL_FLOW))
                   > 0)
                {
                    gst_cuda_of->farneback_flags = g_value_get_int(value);
                    g_assert(properties[PROP_FARNEBACK_FLAGS] != NULL);
                    g_object_notify_by_pspec(
                        gobject, properties[PROP_FARNEBACK_FLAGS]);
                }
                else
                {
                    GST_WARNING_OBJECT(
                        gst_cuda_of,
                        "Could not set the farneback-flags property. The given "
                        "value did not contain either of the valid bit-flags. "
                        "Leaving at the previous value.");
                }
            }
            break;
        case PROP_FARNEBACK_NUMBER_OF_ITERATIONS:
            if(gst_cuda_of->farneback_number_of_iterations
               != g_value_get_int(value))
            {
                gst_cuda_of->farneback_number_of_iterations
                    = g_value_get_int(value);
                g_assert(
                    properties[PROP_FARNEBACK_NUMBER_OF_ITERATIONS] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_FARNEBACK_NUMBER_OF_ITERATIONS]);
            }
            break;
        case PROP_FARNEBACK_NUMBER_OF_LEVELS:
            if(gst_cuda_of->farneback_number_of_levels
               != g_value_get_int(value))
            {
                gst_cuda_of->farneback_number_of_levels
                    = g_value_get_int(value);
                g_assert(properties[PROP_FARNEBACK_NUMBER_OF_LEVELS] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_FARNEBACK_NUMBER_OF_LEVELS]);
            }
            break;
        case PROP_FARNEBACK_POLYNOMIAL_EXPANSION_N:
            if(gst_cuda_of->farneback_polynomial_expansion_n
               != g_value_get_int(value))
            {
                if(g_value_get_int(value) == 5 || g_value_get_int(value) == 7)
                {
                    gst_cuda_of->farneback_polynomial_expansion_n
                        = g_value_get_int(value);
                    g_assert(
                        properties[PROP_FARNEBACK_POLYNOMIAL_EXPANSION_N]
                        != NULL);
                    g_object_notify_by_pspec(
                        gobject,
                        properties[PROP_FARNEBACK_POLYNOMIAL_EXPANSION_N]);
                }
                else
                {
                    GST_WARNING_OBJECT(
                        gst_cuda_of,
                        "Could not set the farneback-polynomial-expansion-n "
                        "property. The given value did not contain either of "
                        "the valid values. Leaving at the previous value.");
                }
            }
            break;
        case PROP_FARNEBACK_POLYNOMIAL_EXPANSION_SIGMA:
            if(gst_cuda_of->farneback_polynomial_expansion_sigma
               != g_value_get_double(value))
            {
                gst_cuda_of->farneback_polynomial_expansion_sigma
                    = g_value_get_double(value);
                g_assert(
                    properties[PROP_FARNEBACK_POLYNOMIAL_EXPANSION_SIGMA]
                    != NULL);
                g_object_notify_by_pspec(
                    gobject,
                    properties[PROP_FARNEBACK_POLYNOMIAL_EXPANSION_SIGMA]);
            }
            break;
        case PROP_FARNEBACK_PYRAMID_SCALE:
            if(gst_cuda_of->farneback_pyramid_scale
               != g_value_get_double(value))
            {
                if(gst_cuda_of->farneback_fast_pyramids == FALSE)
                {
                    gst_cuda_of->farneback_pyramid_scale
                        = g_value_get_double(value);
                    g_assert(properties[PROP_FARNEBACK_PYRAMID_SCALE] != NULL);
                    g_object_notify_by_pspec(
                        gobject, properties[PROP_FARNEBACK_PYRAMID_SCALE]);
                }
                else if(
                    gst_cuda_of->farneback_fast_pyramids == TRUE
                    && G_APPROX_VALUE(g_value_get_double(value), 0.5, 1e-6)
                           == TRUE)
                {
                    gst_cuda_of->farneback_pyramid_scale
                        = g_value_get_double(value);
                    g_assert(properties[PROP_FARNEBACK_PYRAMID_SCALE] != NULL);
                    g_object_notify_by_pspec(
                        gobject, properties[PROP_FARNEBACK_PYRAMID_SCALE]);
                }
                else
                {
                    GST_WARNING_OBJECT(
                        gst_cuda_of,
                        "Could not set the farneback-pyramid-scale property. "
                        "The given value was not set to 0.5 when the "
                        "farneback-fast-pyramids property is set to TRUE. "
                        "Leaving at the previous value.");
                }
            }
            break;
        case PROP_FARNEBACK_WINDOW_SIZE:
            if(gst_cuda_of->farneback_window_size != g_value_get_int(value))
            {
                gst_cuda_of->farneback_window_size = g_value_get_int(value);
                g_assert(properties[PROP_FARNEBACK_WINDOW_SIZE] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_FARNEBACK_WINDOW_SIZE]);
            }
            break;
        case PROP_NVIDIA_ENABLE_COST_BUFFER:
            if(gst_cuda_of->nvidia_enable_cost_buffer
               != g_value_get_boolean(value))
            {
                gst_cuda_of->nvidia_enable_cost_buffer
                    = g_value_get_boolean(value);
                g_assert(properties[PROP_NVIDIA_ENABLE_COST_BUFFER] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_NVIDIA_ENABLE_COST_BUFFER]);
            }
            break;
        case PROP_NVIDIA_ENABLE_EXTERNAL_HINTS:
            if(gst_cuda_of->nvidia_enable_external_hints
               != g_value_get_boolean(value))
            {
                gst_cuda_of->nvidia_enable_external_hints
                    = g_value_get_boolean(value);
                g_assert(properties[PROP_NVIDIA_ENABLE_EXTERNAL_HINTS] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_NVIDIA_ENABLE_EXTERNAL_HINTS]);
            }
            break;
        case PROP_NVIDIA_ENABLE_TEMPORAL_HINTS:
            if(gst_cuda_of->nvidia_enable_temporal_hints
               != g_value_get_boolean(value))
            {
                gst_cuda_of->nvidia_enable_temporal_hints
                    = g_value_get_boolean(value);
                g_assert(properties[PROP_NVIDIA_ENABLE_TEMPORAL_HINTS] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_NVIDIA_ENABLE_TEMPORAL_HINTS]);
            }
            break;
        case PROP_NVIDIA_HINT_VECTOR_GRID_SIZE:
            if(gst_cuda_of->nvidia_hint_vector_grid_size
               != g_value_get_enum(value))
            {
                gst_cuda_of->nvidia_hint_vector_grid_size
                    = g_value_get_enum(value);
                g_assert(properties[PROP_NVIDIA_HINT_VECTOR_GRID_SIZE] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_NVIDIA_HINT_VECTOR_GRID_SIZE]);
            }
            break;
        case PROP_NVIDIA_OUTPUT_VECTOR_GRID_SIZE:
            if(gst_cuda_of->nvidia_output_vector_grid_size
               != g_value_get_enum(value))
            {
                gst_cuda_of->nvidia_output_vector_grid_size
                    = g_value_get_enum(value);
                g_assert(
                    properties[PROP_NVIDIA_OUTPUT_VECTOR_GRID_SIZE] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_NVIDIA_OUTPUT_VECTOR_GRID_SIZE]);
            }
            break;
        case PROP_NVIDIA_PERFORMANCE_PRESET:
            if(gst_cuda_of->nvidia_performance_preset
               != g_value_get_enum(value))
            {
                gst_cuda_of->nvidia_performance_preset
                    = g_value_get_enum(value);
                g_assert(properties[PROP_NVIDIA_PERFORMANCE_PRESET] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_NVIDIA_PERFORMANCE_PRESET]);
            }
            break;
        case PROP_OPTICAL_FLOW_ALGORITHM:
            if(gst_cuda_of->optical_flow_algorithm != g_value_get_enum(value))
            {
                gst_cuda_of->optical_flow_algorithm = g_value_get_enum(value);
                g_assert(properties[PROP_OPTICAL_FLOW_ALGORITHM] != NULL);
                g_object_notify_by_pspec(
                    gobject, properties[PROP_OPTICAL_FLOW_ALGORITHM]);
            }
            break;
        default:
            g_assert_not_reached();
    }
}

static gboolean gst_cuda_of_start(GstBaseTransform *trans)
{
    GstCudaOf *self = GST_CUDA_OF(trans);
    GstCudaOfPrivate *self_private
        = gst_cuda_of_get_instance_private_typesafe(self);
    gboolean result = TRUE;

    result = GST_BASE_TRANSFORM_CLASS(parent_class)->start(trans);

    if(result)
    {
        if(self_private->prev_buffer != NULL)
        {
            gst_buffer_unref(self_private->prev_buffer);
            self_private->prev_buffer = NULL;
        }

        self_private->algorithm_is_initialised = FALSE;

        result = TRUE;
    }

    return result;
}

static gboolean gst_cuda_of_stop(GstBaseTransform *trans)
{
    GstCudaOf *self = GST_CUDA_OF(trans);
    GstCudaOfPrivate *self_private
        = gst_cuda_of_get_instance_private_typesafe(self);
    gboolean result = TRUE;

    if(self_private->prev_buffer != NULL)
    {
        gst_buffer_unref(self_private->prev_buffer);
        self_private->prev_buffer = NULL;
    }

    self_private->algorithm_is_initialised = FALSE;

    self_private->algorithms.dense_optical_flow_algorithm.reset();

    if(!self_private->algorithms.nvidia_optical_flow_algorithm.empty())
    {
        self_private->algorithms.nvidia_optical_flow_algorithm
            ->collectGarbage();
    }
    self_private->algorithms.nvidia_optical_flow_algorithm.reset();

    self_private->algorithms.sparse_optical_flow_algorithm.reset();

    result = GST_BASE_TRANSFORM_CLASS(parent_class)->stop(trans);

    return result;
}

static GstFlowReturn gst_cuda_of_transform(
    GstBaseTransform *trans,
    GstBuffer *inbuf,
    GstBuffer *outbuf)
{
    GstCudaOf *self = GST_CUDA_OF(trans);
    GstCudaOfPrivate *self_private
        = gst_cuda_of_get_instance_private_typesafe(self);
    GstFlowReturn result = GST_FLOW_OK;

    try
    {
        if(!gst_cuda_context_push(self->parent.context))
        {
            throw GstCudaException("Could not push CUDA context.");
        }

        if(!self_private->algorithm_is_initialised)
        {
            gst_cuda_of_init_algorithm(
                self, (GstCudaOfAlgorithm)(self->optical_flow_algorithm));
        }

        if(self_private->prev_buffer != NULL)
        {
            cv::cuda::GpuMat optical_flow_vectors
                = gst_cuda_of_calculate_optical_flow(
                    self, inbuf, self_private->prev_buffer);
            GstMetaOpticalFlow *meta = GST_META_OPTICAL_FLOW_ADD(outbuf);
            meta->optical_flow_vectors
                = new cv::cuda::GpuMat(optical_flow_vectors);
            meta->context
                = GST_CUDA_CONTEXT(gst_object_ref(self->parent.context));
            gst_buffer_unref(self_private->prev_buffer);
        }

        self_private->prev_buffer = gst_buffer_copy(inbuf);

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

    return result;
}

// clang-format off
/******************************************************************************/
// clang-format on
