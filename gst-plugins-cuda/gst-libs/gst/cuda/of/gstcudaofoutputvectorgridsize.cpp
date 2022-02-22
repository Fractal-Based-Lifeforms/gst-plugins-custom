/**************************** Includes and Macros *****************************/

#include <gst/cuda/of/gstcudaofoutputvectorgridsize.h>

#include <glib-object.h>
#include <gst/gst.h>

/**************************** Function Definitions ****************************/

extern GType gst_cuda_of_output_vector_grid_size_get_type()
{
    static GType output_vector_grid_size_type = 0;
    static const GEnumValue output_vector_grid_sizes[]
        = {{OPTICAL_FLOW_OUTPUT_VECTOR_GRID_SIZE_1, "1x1 Grid Size", "1x1"},
           {OPTICAL_FLOW_OUTPUT_VECTOR_GRID_SIZE_2, "2x2 Grid Size", "2x2"},
           {OPTICAL_FLOW_OUTPUT_VECTOR_GRID_SIZE_4, "4x4 Grid Size", "4x4"},
           {0, NULL, NULL}};

    if(g_once_init_enter(&output_vector_grid_size_type))
    {
        GType new_type = g_enum_register_static(
            g_intern_static_string("GstCudaOfOutputVectorGridSize"),
            output_vector_grid_sizes);
        g_once_init_leave(&output_vector_grid_size_type, new_type);
    }

    return output_vector_grid_size_type;
}

