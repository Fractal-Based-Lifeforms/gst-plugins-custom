#include <Poco/Path.h>
#include <gst/gst.h>
#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    GstPlugin *nvcodec_plugin = NULL;
    GError *error = NULL;
    int result = 0;

    if(!gst_is_initialized())
    {
        gst_init(NULL, NULL);
    }

    nvcodec_plugin = gst_plugin_load_file(
        Poco::Path(NVCODEC_PLUGIN_PATH).absolute().toString().c_str(), &error);

    if(nvcodec_plugin == NULL)
    {
        throw std::runtime_error("Could not load the nvcodec plugin");
    }

    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();

    if(nvcodec_plugin != NULL)
    {
        gst_object_unref(nvcodec_plugin);
        nvcodec_plugin = NULL;
    }

    gst_deinit();

    return result;
}
