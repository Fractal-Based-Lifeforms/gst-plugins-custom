sources = files(['gsttranscoder.c', 'gsttranscoder-signal-adapter.c'])
headers = files(['gsttranscoder.h', 'transcoder-prelude.h', 'gsttranscoder-signal-adapter.h'])

transcoder_enums = gnome.mkenums_simple('transcoder-enumtypes',
  sources : headers,
  body_prefix : '#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif',
  header_prefix : '#include <gst/transcoder/transcoder-prelude.h>',
  decorator: 'GST_TRANSCODER_API',
  install_header: false,
  install_dir : join_paths(get_option('includedir'), 'gstreamer-1.0/gst/transcoder'))

gsttranscoder_c = transcoder_enums[0]
gsttranscoder_h = transcoder_enums[1]

transcoder_gen_sources = [gsttranscoder_h]

gst_transcoder = library('gsttranscoder-' + api_version,
  sources + [gsttranscoder_c]  + transcoder_gen_sources,
  install: false,
  include_directories : [configinc, libsinc],
  dependencies: [gst_dep, gstpbutils_dep],
  c_args: gst_plugins_cuda_args + ['-DGST_USE_UNSTABLE_API', '-DBUILDING_GST_TRANSCODER'],
  soversion : soversion,
)

gst_transcoder_dep = declare_dependency(link_with: gst_transcoder,
  dependencies : [gst_dep, gstpbutils_dep],
  include_directories : [libsinc],
  sources: transcoder_gen_sources,
)

meson.override_dependency('gstreamer-transcoder-1.0', gst_transcoder_dep)
