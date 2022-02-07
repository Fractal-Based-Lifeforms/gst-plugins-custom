/* GStreamer
 * Copyright (C) <2018-2019> Seungha Yang <seungha.yang@navercorp.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef __GST_CUDA_UTILS_H__
#define __GST_CUDA_UTILS_H__

#include "gstcudacontext.h"
#include "gstcudaloader.h"
#include <gst/gst.h>

G_BEGIN_DECLS

#ifndef GST_DISABLE_GST_DEBUG
static inline gboolean _gst_cuda_debug(
    CUresult result,
    GstDebugCategory *category,
    const gchar *file,
    const gchar *function,
    gint line)
{
    const gchar *_error_name, *_error_text;
    if(result != CUDA_SUCCESS)
    {
        CuGetErrorName(result, &_error_name);
        CuGetErrorString(result, &_error_text);
        gst_debug_log(
            category,
            GST_LEVEL_WARNING,
            file,
            function,
            line,
            NULL,
            "CUDA call failed: %s, %s",
            _error_name,
            _error_text);

        return FALSE;
    }

    return TRUE;
}

/**
 * gst_cuda_result:
 * @result: CUDA device API return code #CUresult
 *
 * Returns: %TRUE if CUDA device API call result is CUDA_SUCCESS
 */
#define gst_cuda_result(result) \
    _gst_cuda_debug(result, GST_CAT_DEFAULT, __FILE__, GST_FUNCTION, __LINE__)
#else

static inline gboolean _gst_cuda_debug(
    CUresult result,
    GstDebugCategory *category,
    const gchar *file,
    const gchar *function,
    gint line)
{
    return result == CUDA_SUCCESS;
}

/**
 * gst_cuda_result:
 * @result: CUDA device API return code #CUresult
 *
 * Returns: %TRUE if CUDA device API call result is CUDA_SUCCESS
 */
#define gst_cuda_result(result) \
    _gst_cuda_debug(result, NULL, __FILE__, GST_FUNCTION, __LINE__)
#endif

typedef enum
{
    GST_CUDA_QUARK_GRAPHICS_RESOURCE = 0,

    /* end of quark list */
    GST_CUDA_QUARK_MAX = 1
} GstCudaQuarkId;

typedef enum
{
    GST_CUDA_GRAPHICS_RESOURCE_NONE = 0,
    GST_CUDA_GRAPHICS_RESOURCE_GL_BUFFER = 1,
} GstCudaGraphicsResourceType;

typedef struct _GstCudaGraphicsResource
{
    GstCudaContext *cuda_context;
    /* GL context (or d3d11 context in the future) */
    GstObject *graphics_context;

    GstCudaGraphicsResourceType type;
    CUgraphicsResource resource;
    CUgraphicsRegisterFlags flags;

    gboolean registered;
    gboolean mapped;
} GstCudaGraphicsResource;

extern __attribute__((visibility("default"))) gboolean
gst_cuda_ensure_element_context(
    GstElement *element,
    gint device_id,
    GstCudaContext **cuda_ctx);

extern __attribute__((visibility("default"))) gboolean
gst_cuda_handle_set_context(
    GstElement *element,
    GstContext *context,
    gint device_id,
    GstCudaContext **cuda_ctx);

extern __attribute__((visibility("default"))) gboolean
gst_cuda_handle_context_query(
    GstElement *element,
    GstQuery *query,
    GstCudaContext *cuda_ctx);

extern __attribute__((visibility("default"))) GstContext *
gst_context_new_cuda_context(GstCudaContext *context);

extern __attribute__((visibility("default"))) GQuark
gst_cuda_quark_from_id(GstCudaQuarkId id);

extern __attribute__((visibility("default"))) GstCudaGraphicsResource *
gst_cuda_graphics_resource_new(
    GstCudaContext *context,
    GstObject *graphics_context,
    GstCudaGraphicsResourceType type);

extern __attribute__((visibility("default"))) gboolean
gst_cuda_graphics_resource_register_gl_buffer(
    GstCudaGraphicsResource *resource,
    guint buffer,
    CUgraphicsRegisterFlags flags);

extern __attribute__((visibility("default"))) void
gst_cuda_graphics_resource_unregister(GstCudaGraphicsResource *resource);

extern __attribute__((visibility("default"))) CUgraphicsResource
gst_cuda_graphics_resource_map(
    GstCudaGraphicsResource *resource,
    CUstream stream,
    CUgraphicsMapResourceFlags flags);

extern __attribute__((visibility("default"))) void
gst_cuda_graphics_resource_unmap(
    GstCudaGraphicsResource *resource,
    CUstream stream);

extern __attribute__((visibility("default"))) void
gst_cuda_graphics_resource_free(GstCudaGraphicsResource *resource);

G_END_DECLS

#endif /* __GST_CUDA_UTILS_H__ */
