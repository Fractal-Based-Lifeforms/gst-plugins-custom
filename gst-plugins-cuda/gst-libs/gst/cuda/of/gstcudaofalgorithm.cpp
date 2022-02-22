/**************************** Includes and Macros *****************************/

#include <gst/cuda/of/gstcudaofalgorithm.h>

#include <glib-object.h>
#include <gst/gst.h>

/**************************** Function Definitions ****************************/

extern GType gst_cuda_of_algorithm_get_type()
{
    static GType algorithm_type = 0;
    static const GEnumValue algorithms[]
        = {{OPTICAL_FLOW_ALGORITHM_FARNEBACK,
            "Farneback CUDA Optical Flow Algorithm",
            "farneback"},
           {OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0,
            "NVIDIA v1 Hardware Optical Flow Algorithm",
            "nvidia-1.0"},
           {OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0,
            "NVIDIA v2 Hardware Optical Flow Algorithm",
            "nvidia-2.0"},
           {0, NULL, NULL}};

    if(g_once_init_enter(&algorithm_type))
    {
        GType new_type = g_enum_register_static(
            g_intern_static_string("GstCudaOfAlgorithm"), algorithms);
        g_once_init_leave(&algorithm_type, new_type);
    }

    return algorithm_type;
}
