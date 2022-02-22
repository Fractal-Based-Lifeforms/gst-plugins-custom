#ifndef _CUDA_OF_OUTPUT_VECTOR_GRID_SIZE_H_
#define _CUDA_OF_OUTPUT_VECTOR_GRID_SIZE_H_

#include <glib-object.h>
#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_CUDA_OF_OUTPUT_VECTOR_GRID_SIZE \
    (gst_cuda_of_output_vector_grid_size_get_type())

/**
 * \brief An enumeration containing the list of NVIDIA Optical Flow algorithm
 * output vector grid sizes as supported by OpenCV.
 *
 * \notes This is only used for version 2.0 and above of the NVIDIA Optical
 * Flow algorithm. NVIDIA Optical Flow version 1.0 uses a fixed grid-size of
 * 4x4 for both hint vectors and output vectors.
 */
typedef enum _GstCudaOfOutputVectorGridSize
{
    /**
     * \brief The NVIDIA output-vector grid-size for a 1x1 output-vector.
     *
     * \notes The documentation for this is lacking in OpenCV, but basically a
     * 1x1 output-vector grid-size means that each vector in the grid is
     * representative of a single pixel.
     */
    OPTICAL_FLOW_OUTPUT_VECTOR_GRID_SIZE_1 = 1,
    /**
     * \brief The NVIDIA output-vector grid-size for a 2x2 output-vector.
     *
     * \notes The documentation for this is lacking in OpenCV, but basically a
     * 2x2 output-vector grid-size means that each vector in the grid is
     * representative of a 2x2 set of pixels.
     */
    OPTICAL_FLOW_OUTPUT_VECTOR_GRID_SIZE_2 = 2,
    /**
     * \brief The NVIDIA output-vector grid-size for a 4x4 output-vector.
     *
     * \notes The documentation for this is lacking in OpenCV, but basically a
     * 4x4 output-vector grid-size means that each vector in the grid is
     * representative of a 4x4 set of pixels.
     */
    OPTICAL_FLOW_OUTPUT_VECTOR_GRID_SIZE_4 = 4

} GstCudaOfOutputVectorGridSize;

/**
 * \brief Type creation/retrieval function for the GstCudaOfOutputVectorGridSize
 * enum type.
 *
 * \details This function creates and registers the GstCudaOfOutputVectorGridSize
 * enum type for the first invocation. The GType instance for the
 * GstCudaOfOutputVectorGridSize enum type is then returned.
 *
 * \details For subsequent invocations, the GType instance for the
 * GstCudaOfOutputVectorGridSize enum type is returned immediately.
 *
 * \returns A GType instance representing the type information for the
 * GstCudaOfOutputVectorGridSize enum type.
 */
extern __attribute__((visibility("default"))) GType
gst_cuda_of_output_vector_grid_size_get_type();

G_END_DECLS

#endif

