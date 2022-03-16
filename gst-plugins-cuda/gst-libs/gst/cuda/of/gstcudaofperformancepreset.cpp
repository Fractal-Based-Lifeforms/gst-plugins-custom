/**************************** Includes and Macros *****************************/

#include <gst/cuda/of/gstcudaofperformancepreset.h>

#include <glib-object.h>
#include <gst/gst.h>

/**************************** Function Definitions ****************************/

extern GType gst_cuda_of_performance_preset_get_type()
{
    static GType performance_preset_type = 0;
    static const GEnumValue performance_presets[]
        = {{OPTICAL_FLOW_PERFORMANCE_PRESET_SLOW,
            "Slow performance preset",
            "slow"},
           {OPTICAL_FLOW_PERFORMANCE_PRESET_MEDIUM,
            "Medium performance preset",
            "medium"},
           {OPTICAL_FLOW_PERFORMANCE_PRESET_FAST,
            "Fast performance preset",
            "fast"},
           {0, NULL, NULL}};

    if(g_once_init_enter(&performance_preset_type))
    {
        GType new_type = g_enum_register_static(
            g_intern_static_string("GstCudaOfPerformancePreset"),
            performance_presets);
        g_once_init_leave(&performance_preset_type, new_type);
    }

    return performance_preset_type;
}
