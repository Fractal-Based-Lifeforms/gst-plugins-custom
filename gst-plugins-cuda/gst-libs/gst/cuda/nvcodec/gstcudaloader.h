/* GStreamer
 * Copyright (C) 2019 Seungha Yang <seungha.yang@navercorp.com>
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

#ifndef __GST_CUDA_LOADER_H__
#define __GST_CUDA_LOADER_H__

#include <gst/cuda/stub/cuda.h>

#include <gst/gst.h>

G_BEGIN_DECLS

extern __attribute__((visibility("default"))) gboolean
gst_cuda_load_library(void);

/* cuda.h */
extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuInit(unsigned int Flags);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGetErrorName(CUresult error, const char **pStr);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGetErrorString(CUresult error, const char **pStr);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuCtxDestroy(CUcontext ctx);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuCtxPopCurrent(CUcontext *pctx);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuCtxPushCurrent(CUcontext ctx);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuCtxDisablePeerAccess(CUcontext peerContext);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsMapResources(
    unsigned int count,
    CUgraphicsResource *resources,
    CUstream hStream);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsUnmapResources(
    unsigned int count,
    CUgraphicsResource *resources,
    CUstream hStream);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsSubResourceGetMappedArray(
    CUarray *pArray,
    CUgraphicsResource resource,
    unsigned int arrayIndex,
    unsigned int mipLevel);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsResourceGetMappedPointer(
    CUdeviceptr *pDevPtr,
    size_t *pSize,
    CUgraphicsResource resource);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsUnregisterResource(CUgraphicsResource resource);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuMemAlloc(CUdeviceptr *dptr, unsigned int bytesize);

extern __attribute__((visibility("default"))) CUresult CUDAAPI CuMemAllocPitch(
    CUdeviceptr *dptr,
    size_t *pPitch,
    size_t WidthInBytes,
    size_t Height,
    unsigned int ElementSizeBytes);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuMemAllocHost(void **pp, unsigned int bytesize);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuMemcpy2D(const CUDA_MEMCPY2D *pCopy);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuMemFree(CUdeviceptr dptr);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuMemFreeHost(void *p);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuStreamCreate(CUstream *phStream, unsigned int Flags);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuStreamDestroy(CUstream hStream);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuStreamSynchronize(CUstream hStream);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuDeviceGet(CUdevice *device, int ordinal);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuDeviceGetCount(int *count);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuDeviceGetName(char *name, int len, CUdevice dev);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuDriverGetVersion(int *driverVersion);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuModuleLoadData(CUmodule *module, const void *image);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuModuleUnload(CUmodule module);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuTexObjectCreate(
    CUtexObject *pTexObject,
    const CUDA_RESOURCE_DESC *pResDesc,
    const CUDA_TEXTURE_DESC *pTexDesc,
    const CUDA_RESOURCE_VIEW_DESC *pResViewDesc);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuTexObjectDestroy(CUtexObject texObject);

extern __attribute__((visibility("default"))) CUresult CUDAAPI CuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra);

/* cudaGL.h */
extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsGLRegisterImage(
    CUgraphicsResource *pCudaResource,
    unsigned int image,
    unsigned int target,
    unsigned int Flags);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsGLRegisterBuffer(
    CUgraphicsResource *pCudaResource,
    unsigned int buffer,
    unsigned int Flags);

extern __attribute__((visibility("default"))) CUresult CUDAAPI
CuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags);

extern __attribute__((visibility("default"))) CUresult CUDAAPI CuGLGetDevices(
    unsigned int *pCudaDeviceCount,
    CUdevice *pCudaDevices,
    unsigned int cudaDeviceCount,
    CUGLDeviceList deviceList);

G_END_DECLS
#endif /* __GST_CUDA_LOADER_H__ */
