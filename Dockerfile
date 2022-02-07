# This is the common build environment for Meson-based projects such as
# GStreamer and our gstreamer-plugins repository.
#
# Separating this into a separate build target lets us re-use this to leverage
# the cache to provide a consistent build environment.
# - J.O.
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04 AS meson-build-environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        bison \
        ca-certificates \
        flex \
        git \
        pkg-config \
        python3 \
        python3-pip \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*
RUN pip install --no-cache-dir \
    meson==0.60.2 \
    ninja==1.10.2.3

# This is the common build environment for CMake-based projects such as
# OpenCV.
#
# Separating this into a separate build target lets us re-use this to leverage
# the cache to provide a consistent build environment.
# - J.O.
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04 AS cmake-build-environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gpg \
        wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | tee  /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" \
        | tee  /etc/apt/sources.list.d/kitware.list > /dev/null \
    && apt-get update \
    && rm /usr/share/keyrings/kitware-archive-keyring.gpg \
    && apt-get install -y --no-install-recommends \
        kitware-archive-keyring \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# This is the build target for the GStreamer installation. For now, it builds
# GStreamer 1.19.3 by default, but it may be changed to build GStreamer 1.20.x
# once that has been made available.
# - J.O.
FROM meson-build-environment AS gstreamer-build
ENV DEBIAN_FRONTEND=noninteractive
ARG GSTREAMER_TAG=1.19.3
RUN mkdir -p /opt/gstreamer \
    && git clone \
        https://gitlab.freedesktop.org/gstreamer/gstreamer \
        --depth=1 \
        --branch=$GSTREAMER_TAG \
    && cd gstreamer \
    && meson builddir \
    && ninja -C builddir \
    && DESTDIR=/opt/gstreamer/ ninja install -C builddir \
    && cd .. \
    && rm -rf gstreamer

# This is the build target for the OpenCV installation. For now, it builds
# OpenCV 4.5.4 with the minimum number of modules to support CUDA-based optical
# flow.
# - J.O.
FROM cmake-build-environment AS opencv-build
ARG OPENCV_TAG=4.5.4
RUN mkdir -p /opt/opencv \
    && git clone \
        https://github.com/opencv/opencv \
        --depth=1 \
        --branch=$OPENCV_TAG \
    && git clone \
        https://github.com/opencv/opencv_contrib \
        --depth=1 \
        --branch=$OPENCV_TAG \
    && cd opencv \
    && mkdir -p cmake-build \
    && cd cmake-build \
    && cmake \
        -DBUILD_DOCS=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_opencv_alphamat=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_aruco=OFF \
        -DBUILD_opencv_barcode=OFF \
        -DBUILD_opencv_bgsegm=OFF \
        -DBUILD_opencv_bioinspired=OFF \
        -DBUILD_opencv_calib3d=ON \
        -DBUILD_opencv_ccalib=OFF \
        -DBUILD_opencv_cudaarithm=ON \
        -DBUILD_opencv_cudabgsegm=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DBUILD_opencv_cudafeatures2d=OFF \
        -DBUILD_opencv_cudafilters=OFF \
        -DBUILD_opencv_cudaimgproc=ON \
        -DBUILD_opencv_cudalegacy=OFF \
        -DBUILD_opencv_cudaobjdetect=OFF \
        -DBUILD_opencv_cudaoptflow=ON \
        -DBUILD_opencv_cudastereo=OFF \
        -DBUILD_opencv_cudawarping=ON \
        -DBUILD_opencv_cudev=ON \
        -DBUILD_opencv_cvv=OFF \
        -DBUILD_opencv_datasets=OFF \
        -DBUILD_opencv_dnn_objdetect=OFF \
        -DBUILD_opencv_dnn_superres=OFF \
        -DBUILD_opencv_dnn=OFF \
        -DBUILD_opencv_dpm=OFF \
        -DBUILD_opencv_face=OFF \
        -DBUILD_opencv_features2d=ON \
        -DBUILD_opencv_flann=ON \
        -DBUILD_opencv_freetype=OFF \
        -DBUILD_opencv_fuzzy=OFF \
        -DBUILD_opencv_gapi=OFF \
        -DBUILD_opencv_hdf=OFF \
        -DBUILD_opencv_hfs=OFF \
        -DBUILD_opencv_highgui=OFF \
        -DBUILD_opencv_img_hash=OFF \
        -DBUILD_opencv_imgcodecs=ON \
        -DBUILD_opencv_imgproc=ON \
        -DBUILD_opencv_intensity_transform=OFF \
        -DBUILD_opencv_java_bindings_generator=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_js_bindings_generator=OFF \
        -DBUILD_opencv_js=OFF \
        -DBUILD_opencv_julia=OFF \
        -DBUILD_opencv_line_descriptor=OFF \
        -DBUILD_opencv_matlab=OFF \
        -DBUILD_opencv_mcc=OFF \
        -DBUILD_opencv_ml=OFF \
        -DBUILD_opencv_objc_bindings_generator=OFF \
        -DBUILD_opencv_objdetect=OFF \
        -DBUILD_opencv_optflow=ON \
        -DBUILD_opencv_ovis=OFF \
        -DBUILD_opencv_phase_unwrapping=OFF \
        -DBUILD_opencv_photo=OFF \
        -DBUILD_opencv_plot=OFF \
        -DBUILD_opencv_python_bindings_generator=OFF \
        -DBUILD_opencv_python_tests=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_quality=OFF \
        -DBUILD_opencv_rapid=OFF \
        -DBUILD_opencv_reg=OFF \
        -DBUILD_opencv_rgbd=OFF \
        -DBUILD_opencv_saliency=OFF \
        -DBUILD_opencv_sfm=OFF \
        -DBUILD_opencv_shape=OFF \
        -DBUILD_opencv_stereo=OFF \
        -DBUILD_opencv_stitching=OFF \
        -DBUILD_opencv_structured_light=OFF \
        -DBUILD_opencv_superres=OFF \
        -DBUILD_opencv_surface_matching=OFF \
        -DBUILD_opencv_text=OFF \
        -DBUILD_opencv_tracking=OFF \
        -DBUILD_opencv_ts=OFF \
        -DBUILD_opencv_video=ON \
        -DBUILD_opencv_videoio=OFF \
        -DBUILD_opencv_videostab=OFF \
        -DBUILD_opencv_viz=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DBUILD_opencv_xfeatures2d=OFF \
        -DBUILD_opencv_ximgproc=ON \
        -DBUILD_opencv_xobjdetect=OFF \
        -DBUILD_opencv_xphoto=OFF \
        -DBUILD_OPENEXR=OFF \
        -DBUILD_OPENJPEG=OFF \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_WEBP=OFF \
        -DCMAKE_BUILD_TYPE=Release  \
        -DINSTALL_CREATE_DISTRIB=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DOPENCV_FORCE_3RDPARTY_BUILD=ON \
        -DWITH_1394=OFF \
        -DWITH_CUBLAS=OFF \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=OFF \
        -DWITH_CUFFT=OFF \
        -DWITH_DIRECTX=OFF \
        -DWITH_DSHOW=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GTK=OFF \
        -DWITH_IMGCODEC_HDR=OFF \
        -DWITH_IMGCODEC_PFM=OFF \
        -DWITH_IMGCODEC_PXM=OFF \
        -DWITH_IMGCODEC_SUNRASTER=OFF \
        -DWITH_INF_ENGINE=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_JPEG=OFF \
        -DWITH_LAPACK=OFF \
        -DWITH_MSMF=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENCLAMDBLAS=OFF \
        -DWITH_OPENCLAMDFFT=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_PNG=OFF \
        -DWITH_PROTOBUF=OFF \
        -DWITH_QUIRC=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_V4L=OFF \
        -DWITH_VTK=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_WIN32UI=OFF \
        .. \
    && make -j`nproc --all` \
    && DESTDIR=/opt/opencv make install -j`nproc --all`

# This is the build target for the gstreamer-plugins installation. For now, it
# builds our custom CUDA-based plugins for installation in a later build
# target.
# - J.O.
FROM meson-build-environment AS gstreamer-plugins-build
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        rapidjson-dev \
    && apt-get clean \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*
COPY --from=gstreamer-build /opt/gstreamer /
COPY --from=opencv-build /opt/opencv /
WORKDIR /work
COPY . /work
WORKDIR /work/gst-plugins-cuda
RUN mkdir -p /opt/gstreamer-plugins \
    && meson build -Dtests=disabled \
    && ninja -C build \
    && DESTDIR=/opt/gstreamer-plugins ninja install -C build

# This is the deploy target for the gstreamer-plugins installation. It copies
# the builds for OpenCV 4.5.4, GStreamer 1.19.3 and gstreamer-plugins into
# /usr/local.
# - J.O.
FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04 AS gstreamer-plugins-deploy
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
# This environment variable needs to be set, as 'libgstnvcodec' is built to
# look for 'libnvrtc.so' and not the one in the runtime image.
# - J.O.
ENV GST_NVCODEC_NVRTC_LIBNAME=/usr/local/cuda-11.4/targets/x86_64-linux/lib/libnvrtc.so.11.4.120
COPY --from=gstreamer-build /opt/gstreamer /
COPY --from=opencv-build /opt/opencv /
COPY --from=gstreamer-plugins-build /opt/gstreamer-plugins /
RUN ldconfig
