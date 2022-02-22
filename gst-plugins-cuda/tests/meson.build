if not get_option('tests').disabled() and gtest_dep.found()
  extra_c_args = [
    '-DGST_USE_UNSTABLE_API',
    '-DROOT_DATA_DIRECTORY="' + meson.current_source_dir() + '/data"',
    '-DNVCODEC_PLUGIN_PATH="' + meson.current_build_dir() + '/../sys/nvcodec/libgstnvcodec.so"'
  ]
  extra_cpp_args = [
    '-DGST_USE_UNSTABLE_API',
    '-DROOT_DATA_DIRECTORY="' + meson.current_source_dir() + '/data"',
    '-DNVCODEC_PLUGIN_PATH="' + meson.current_build_dir() + '/../sys/nvcodec/libgstnvcodec.so"'
  ]

  libpthread = cc.find_library('pthread', required: true)
  libdl = cc.find_library('dl', required: true)
  librt = cc.find_library('rt', required: true)
  
  unittest_sources = [
  'src/CUDAFeaturesCell_UnitTest.cpp',
  'src/CUDAFeaturesMatrix_UnitTest.cpp',
  'src/GstCudaFeatureExtractor_UnitTest.cpp',
  'src/GstCudaOf_UnitTest.cpp',
  'src/UnitTests.cpp',
  ]

  test_executable = executable('gst-plugins-custom_unit-test',
    unittest_sources,
    c_args : gst_plugins_cuda_args + extra_c_args,
    cpp_args : gst_plugins_cuda_args + extra_cpp_args,
    include_directories: [configinc, '../sys/nvcodec/nvcodec', '../sys/nvcodec/cudaof'],
    dependencies: [glib_dep, gst_dep, gstbase_dep, gstapp_dep, opencv_dep, poco_dep, libpthread, libdl, librt, gtest_dep, gst_cuda_dep],
    install : false
  )

  test('gst-plugins-custom_unit-test', test_executable)
endif
