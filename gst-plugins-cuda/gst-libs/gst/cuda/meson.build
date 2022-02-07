gst_cuda_sources = files([
  'of/gstmetaopticalflow.cpp',
  'nvcodec/cuda-converter.c',
  'nvcodec/gstcudabasefilter.c',
  'nvcodec/gstcudabasetransform.c',
  'nvcodec/gstcudabufferpool.c',
  'nvcodec/gstcudacontext.c',
  'nvcodec/gstcudaloader.c',
  'nvcodec/gstcudamemory.c',
  'nvcodec/gstcudanvrtc.c',
  'nvcodec/gstcudautils.c',
  'nvcodec/gstnvrtcloader.c',
  'featureextractor/cudafeaturescell.c',
  'featureextractor/cudafeaturesmatrix.c',
  'featureextractor/gstmetaalgorithmfeatures.c',
])

gst_cuda_featureextractor_headers = files([
  'of/gstmetaopticalflow.h',
  'featureextractor/cudafeaturescell.h',
  'featureextractor/cudafeaturesmatrix.h',
  'featureextractor/gstmetaalgorithmfeatures.h',
])
gst_cuda_nvcodec_headers = files([
  'nvcodec/cuda-converter.h',
  'nvcodec/gstcudabasefilter.h',
  'nvcodec/gstcudabasetransform.h',
  'nvcodec/gstcudabufferpool.h',
  'nvcodec/gstcudacontext.h',
  'nvcodec/gstcudaloader.h',
  'nvcodec/gstcudamemory.h',
  'nvcodec/gstcudanvrtc.h',
  'nvcodec/gstcudautils.h',
  'nvcodec/gstnvrtcloader.h',
])
gst_cuda_of_headers = files([
  'of/gstmetaopticalflow.h',
  'featureextractor/cudafeaturescell.h',
  'featureextractor/cudafeaturesmatrix.h',
  'featureextractor/gstmetaalgorithmfeatures.h',
])

gst_cuda_stub_headers = files([
  'stub/cuda.h',
  'stub/nvrtc.h',
])

if get_option('cuda').disabled()
  subdir_done()
endif

install_headers(gst_cuda_featureextractor_headers, subdir : 'gstreamer-1.0/gst/cuda/featureextractor/')
install_headers(gst_cuda_nvcodec_headers, subdir : 'gstreamer-1.0/gst/cuda/nvcodec/')
install_headers(gst_cuda_of_headers, subdir : 'gstreamer-1.0/gst/cuda/of/')
install_headers(gst_cuda_stub_headers, subdir : 'gstreamer-1.0/gst/cuda/stub/')

gst_cuda_incdirs = [
  configinc,
  include_directories('./of'),
  include_directories('./featureextractor'),
  include_directories('./nvcodec'),
  include_directories('./stub')
]
extra_c_args = ['-DGST_USE_UNSTABLE_API']
extra_cpp_args = [
  '-std=gnu++17',
  '-DGST_USE_UNSTABLE_API',
]

if gstgl_dep.found()
  extra_c_args += ['-DHAVE_NVCODEC_GST_GL=1']
  extra_cpp_args += ['-DHAVE_NVCODEC_GST_GL=1']
endif

gst_cuda_dependencies = [
  gst_dep,
  gstbase_dep,
  gstvideo_dep,
  gmodule_dep,
  opencv_dep,
]

gst_cuda = library('gstcuda-' + api_version,
  gst_cuda_sources,
  c_args : gst_plugins_cuda_args,
  cpp_args : gst_plugins_cuda_args,
  include_directories : [configinc, libsinc, gst_cuda_incdirs],
  version : libversion,
  soversion : soversion,
  install : true,
  dependencies : gst_cuda_dependencies
)

pkg_name = 'gstreamer-cuda-1.0'
pkgconfig.generate(gst_cuda,
  libraries : [gst_dep],
  variables : pkgconfig_variables,
  subdirs : pkgconfig_subdirs,
  name : pkg_name,
  description : 'Helper functions and base classes for CUDA integration',
)

gst_cuda_dep = declare_dependency(link_with: gst_cuda,
  include_directories : [libsinc, gst_cuda_incdirs],
  dependencies : gst_cuda_dependencies
)
