gst-plugins-cuda
----------------

This is a fork of the [gst-plugins-bad](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/tree/main/subprojects/gst-plugins-bad) subproject of the GStreamer project. 

In particular, this fork requires:
- The `codecparsers` library.
- The `codecs` library.
- The `transcoder` library.

And expands upon the `nvcodec` plugin, which features a suite of elements that introduce encoding/decoding via the dedicated hardware support featured on NVIDIA GPU cards.

The `nvcodec` plugin has recently (as of 1.19.x) introduced support for buffers held within GPU memory that are managed via the CUDA driver API. This support also includes serveral elements such as `cudadownload`/`cudaupload` and `cudascale` in order to interact and alter buffers held within GPU memory. 

The `cudaof` element uses this newly introduced CUDA/GPU buffer support in order to perform optical flow analysis. The element wraps the GPU buffers in OpenCV GpuMat instances, which are then passed to one of the three supported GPU-based optical flow algorithms (Farneback, NVIDIA Optical Flow v1 and NVIDIA Optical Flow v2). This ensures a higher degree of performance compared to using CPU-based optical flow algorithms, or attempting to transfer the buffers from host memory to GPU memory.
