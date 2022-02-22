#ifndef _CUDA_OF_PERFORMANCE_PRESET_H_
#define _CUDA_OF_PERFORMANCE_PRESET_H_

#include <glib-object.h>
#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_CUDA_OF_PERFORMANCE_PRESET \
    (gst_cuda_of_performance_preset_get_type())

/**
 * \brief An enumeration containing the list of NVIDIA Optical Flow algorithm
 * performance presets as supported by OpenCV.
 */
typedef enum _GstCudaOfPerformancePreset
{
    /**
     * \brief The NVIDIA performance preset for slow performance.
     *
     * \notes This results in significantly slower performance compared to the
     * fast preset, and moderately slower performance compared to the medimum
     * preset, but gives the highest accuracy out of the three available
     * presets.
     */
    OPTICAL_FLOW_PERFORMANCE_PRESET_SLOW = 5,
    /**
     * \brief The NVIDIA performance preset for medimum performance.
     *
     * \notes This results in significantly slower performance compared to the
     * fast preset, and moderately faster performance compared to the slow
     * preset, but gives significantly better accuracy than the fast preset.
     */
    OPTICAL_FLOW_PERFORMANCE_PRESET_MEDIUM = 10,
    /**
     * \brief The NVIDIA performance preset for fast performance.
     *
     * \notes This results in significantly faster performance compared to the
     * slow and medium presets, but has the lowest accuracy out of the three
     * available presets.
     */
    OPTICAL_FLOW_PERFORMANCE_PRESET_FAST = 20
} GstCudaOfPerformancePreset;

/**
 * \brief Type creation/retrieval function for the GstCudaOfPerformancePreset
 * enum type.
 *
 * \details This function creates and registers the GstCudaOfPerformancePreset
 * enum type for the first invocation. The GType instance for the
 * GstCudaOfPerformancePreset enum type is then returned.
 *
 * \details For subsequent invocations, the GType instance for the
 * GstCudaOfPerformancePreset enum type is returned immediately.
 *
 * \returns A GType instance representing the type information for the
 * GstCudaOfPerformancePreset enum type.
 */
extern __attribute__((visibility("default"))) GType
gst_cuda_of_performance_preset_get_type();

G_END_DECLS

#endif

