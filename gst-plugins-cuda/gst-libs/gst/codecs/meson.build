codecs_sources = files([
  'gsth264decoder.c',
  'gsth264picture.c',
  'gsth265decoder.c',
  'gsth265picture.c',
  'gstvp9decoder.c',
  'gstvp9picture.c',
  'gstvp8decoder.c',
  'gstvp8picture.c',
  'gstmpeg2decoder.c',
  'gstmpeg2picture.c',
  'gstav1decoder.c',
  'gstav1picture.c',
  'gstvp9statefulparser.c',
])

codecs_headers = [
  'gsth264decoder.h',
  'gsth264picture.h',
  'gsth265decoder.h',
  'gsth265picture.h',
  'gstvp9decoder.h',
  'gstvp9picture.h',
  'gstvp8decoder.h',
  'gstvp8picture.h',
  'gstmpeg2decoder.h',
  'gstmpeg2picture.h',
  'gstav1decoder.h',
  'gstav1picture.h',
  'gstvp9statefulparser.h',
]

cp_args = [
  '-DGST_USE_UNSTABLE_API',
  '-DBUILDING_GST_CODECS',
]

gstcodecs = library('gstcodecs-' + api_version,
  codecs_sources,
  c_args : gst_plugins_cuda_args + cp_args,
  include_directories : [configinc, libsinc],
  version : libversion,
  soversion : soversion,
  darwin_versions : osxversion,
  install : false,
  dependencies : [gstvideo_dep, gstcodecparsers_dep],
)

gen_sources = []

gstcodecs_dep = declare_dependency(link_with : gstcodecs,
  include_directories : [libsinc],
  sources: gen_sources,
  dependencies : [gstvideo_dep, gstcodecparsers_dep])
