nvcodec_sources = [
  './cudaof/gstcudaof.cpp',
  './cudafeatureextractor/gstcudafeatureextractor.cpp',
  './nvcodec/gstcudaconvert.c',
  './nvcodec/gstcudadownload.c',
  './nvcodec/gstcudafilter.c',
  './nvcodec/gstcudascale.c',
  './nvcodec/gstcudaupload.c',
  './nvcodec/gstcuvidloader.c',
  './nvcodec/gstnvbaseenc.c',
  './nvcodec/gstnvdec.c',
  './nvcodec/gstnvdecoder.c',
  './nvcodec/gstnvenc.c',
  './nvcodec/gstnvh264dec.c',
  './nvcodec/gstnvh264enc.c',
  './nvcodec/gstnvh265dec.c',
  './nvcodec/gstnvh265enc.c',
  './nvcodec/gstnvvp8dec.c',
  './nvcodec/gstnvvp9dec.c',
  'plugin.c',
]

if get_option('nvcodec').disabled()
  subdir_done()
endif

plugin_incdirs = [
  configinc,
]
extra_c_args = ['-DGST_USE_UNSTABLE_API']
cuda_feature_extractor_kernel_source_path = join_paths(get_option('prefix'), data_install_dir, 'CUDA', 'cudafeatureextractorkernels.cu')
extra_cpp_args = [
  '-std=gnu++17',
  '-DGST_USE_UNSTABLE_API',
  '-DGST_CUDA_FEATURE_EXTRACTOR_KERNEL_SOURCE_PATH="' + cuda_feature_extractor_kernel_source_path  + '"',
]

if gstgl_dep.found()
  extra_c_args += ['-DHAVE_NVCODEC_GST_GL=1']
  extra_cpp_args += ['-DHAVE_NVCODEC_GST_GL=1']
endif

nvcodec_dependencies = [
  gstbase_dep,
  gstvideo_dep,
  gstpbutils_dep,
  gstgl_dep,
  gstglproto_dep,
  gmodule_dep,
  gstcodecs_dep,
  opencv_dep,
  rapidjson_dep,
  gst_cuda_dep
]

if cxx.get_id() == 'gcc'
  if cxx.version().version_compare('<9.0.0')
    libstdcxxfs = cxx.find_library('stdc++fs', required: true)
    nvcodec_dependencies += [libstdcxxfs]
  endif
endif

gstnvcodec = library('gstnvcodec',
  nvcodec_sources,
  c_args : gst_plugins_cuda_args + extra_c_args,
  cpp_args : gst_plugins_cuda_args + extra_cpp_args,
  include_directories : plugin_incdirs,
  dependencies : nvcodec_dependencies,
  install : true,
  install_dir : plugins_install_dir,
)
install_data('./cudafeatureextractor/cudafeatureextractorkernels.cu',
  install_dir : join_paths(data_install_dir, 'CUDA'))

plugins += [gstnvcodec]
