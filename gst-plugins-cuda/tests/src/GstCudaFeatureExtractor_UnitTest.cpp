#include <cstdint>
#include <fstream>
#include <gtest/gtest_pred_impl.h>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <thread>

#include <Poco/Path.h>
#include <gst/app/gstappsink.h>
#include <gst/cuda/featureextractor/gstmetaalgorithmfeatures.h>
#include <gst/cuda/of/gstcudaofalgorithm.h>
#include <gst/cuda/of/gstmetaopticalflow.h>
#include <gst/gst.h>
#include <gst/gstbus.h>
#include <gst/gstcaps.h>
#include <gst/gstcapsfeatures.h>
#include <gst/gstmessage.h>
#include <gst/gstpipeline.h>
#include <gst/video/video-format.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "gstcudaof.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"

using ::testing::Values;

namespace
{
    constexpr auto default_cuda_device_id = 0u;
    const cv::Size_<size_t> default_features_matrix_size(20u, 20u);
    const Poco::Path default_frames_path
        = Poco::Path(std::string(ROOT_DATA_DIRECTORY) + std::string("/frames/"))
              .absolute();
    constexpr auto default_magnitude_quadrant_threshold_squared = 2.25f;
    constexpr auto features_per_aggregation = 10u;
    const std::string kernel_source_location
        = Poco::Path(std::string(GST_CUDA_FEATURE_EXTRACTOR_KERNEL_SOURCE_PATH))
              .absolute()
              .toString();

    const cv::Size_<size_t> test_frame_size(1920u, 1080u);
    constexpr auto test_framerate = 5u;

    template<typename T>
    T ceil_div_int(T value, T divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    class TestFeatureExtractorPipeline
    {
        private:
        GstCudaOfAlgorithm _algorithm_type = OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0;
        guint _bus_watch_id = 0u;
        std::size_t _frame_width = 0u;
        std::size_t _frame_height = 0u;
        std::uint32_t _framerate = 0u;
        GMainLoop *_loop = nullptr;
        GstPipeline *_pipeline = nullptr;
        std::shared_ptr<std::queue<GstSample *>> _sample_queue = nullptr;

        public:
        TestFeatureExtractorPipeline(
            const std::size_t frame_width,
            const std::size_t frame_height,
            const std::uint32_t framerate,
            const GstCudaOfAlgorithm algorithm_type,
            const std::shared_ptr<std::queue<GstSample *>> sample_queue)
        {
            this->_bus_watch_id = 0u;
            this->_loop = nullptr;
            this->_pipeline = nullptr;

            this->_frame_width = frame_width;
            this->_frame_height = frame_height;
            this->_framerate = framerate;
            this->_algorithm_type = algorithm_type;

            this->_sample_queue = sample_queue;

            this->CreatePipeline();
        }
        ~TestFeatureExtractorPipeline()
        {
            this->TeardownPipeline();
        }

        void Run()
        {
            this->RunPipeline();
        }

        private:
        static gboolean
        MonitorMessageBus(GstBus *bus, GstMessage *message, gpointer user_data)
        {
            auto *that = static_cast<TestFeatureExtractorPipeline *>(user_data);

            if(GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS
               || GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR)
            {
                if(g_main_loop_is_running(that->_loop))
                {
                    g_main_loop_quit(that->_loop);
                }

                that->_bus_watch_id = 0u;
                return FALSE;
            }

            return TRUE;
        }
        void CreatePipeline()
        {
            this->_pipeline = GST_PIPELINE(gst_parse_launch(
                "multifilesrc name=multifilesrc0 ! "
                "rawvideoparse name=rawvideoparse0 ! "
                "cudaupload ! "
                "cudaof name=cudaof0 ! "
                "cudafeatureextractor name=cudafeatureextractor0 ! "
                "appsink name=appsink0",
                NULL));

            GstElement *multifilesrc = gst_bin_get_by_name(
                GST_BIN(this->_pipeline), "multifilesrc0");

            // clang-format off
            GstCaps *multifilesrc_caps = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "NV12",
                "framerate", GST_TYPE_FRACTION, this->_framerate, 1u,
                "height", G_TYPE_INT, this->_frame_height,
                "width", G_TYPE_INT, this->_frame_width,
                NULL);
            // clang-format on

            // clang-format off
            g_object_set(
                GST_OBJECT(multifilesrc),
                "caps", multifilesrc_caps,
                "location", Poco::Path(
                    default_frames_path, "sample_1080p_h264.%04d.raw")
                    .absolute()
                    .toString()
                    .c_str(),
                "start-index", 1u,
                "stop-index", 2u,
                NULL);
            // clang-format on

            gst_caps_unref(multifilesrc_caps);
            multifilesrc_caps = NULL;

            gst_object_unref(multifilesrc);
            multifilesrc = NULL;

            GstElement *rawvideoparse = gst_bin_get_by_name(
                GST_BIN(this->_pipeline), "rawvideoparse0");

            // clang-format off
            g_object_set(
                GST_OBJECT(rawvideoparse),
                "format",  GST_VIDEO_FORMAT_NV12,
                "framerate", this->_framerate, 1u,
                "height", this->_frame_height,
                "width", this->_frame_width,
                NULL
            );
            // clang-format on

            gst_object_unref(rawvideoparse);
            rawvideoparse = NULL;

            GstElement *cudaof
                = gst_bin_get_by_name(GST_BIN(this->_pipeline), "cudaof0");

            // clang-format off
            g_object_set(
                GST_OBJECT(cudaof),
                "cuda-device-id", default_cuda_device_id,
                "optical-flow-algorithm", this->_algorithm_type,
                NULL
            );
            // clang-format on

            gst_object_unref(cudaof);

            GstElement *cudafeatureextractor = gst_bin_get_by_name(
                GST_BIN(this->_pipeline), "cudafeatureextractor0");

            // clang-format off
            g_object_set(
                GST_OBJECT(cudafeatureextractor),
                "cuda-device-id", default_cuda_device_id,
                "features-matrix-width", default_features_matrix_size.width,
                "features-matrix-height", default_features_matrix_size.height,
                "kernel-source-location", kernel_source_location.c_str(),
                NULL
            );
            // clang-format on

            gst_object_unref(cudafeatureextractor);

            GstElement *appsink
                = gst_bin_get_by_name(GST_BIN(this->_pipeline), "appsink0");

            // clang-format off
            g_object_set(
                GST_OBJECT(appsink),
                "emit-signals", FALSE,
                "wait-on-eos", TRUE,
                NULL
            );
            // clang-format on

            gst_object_unref(appsink);
            appsink = NULL;
        }

        void SampleThread()
        {
            GstAppSink *appsink = GST_APP_SINK(
                gst_bin_get_by_name(GST_BIN(this->_pipeline), "appsink0"));

            while(!gst_app_sink_is_eos(appsink))
            {
                GstSample *sample = gst_app_sink_pull_sample(appsink);

                if(sample != NULL)
                {
                    GstSample *queue_sample = gst_sample_copy(sample);
                    if(queue_sample != NULL)
                    {
                        this->_sample_queue->push(gst_sample_ref(queue_sample));
                        gst_sample_unref(queue_sample);
                    }

                    gst_sample_unref(sample);
                }
            }

            gst_object_unref(GST_OBJECT(appsink));
        }

        void RunPipeline()
        {
            this->_loop = g_main_loop_new(NULL, FALSE);

            GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(this->_pipeline));
            this->_bus_watch_id = gst_bus_add_watch(
                GST_BUS(bus),
                static_cast<GstBusFunc>(
                    TestFeatureExtractorPipeline::MonitorMessageBus),
                static_cast<gpointer>(this));

            gst_element_set_state(
                GST_ELEMENT(this->_pipeline), GST_STATE_PLAYING);

            std::thread sample_thread = std::thread(
                &TestFeatureExtractorPipeline::SampleThread, this);

            g_main_loop_run(this->_loop);

            gst_element_set_state(GST_ELEMENT(this->_pipeline), GST_STATE_NULL);

            if(this->_bus_watch_id != 0u)
            {
                gst_bus_remove_watch(bus);
            }
            gst_object_unref(bus);

            g_main_loop_unref(this->_loop);

            if(sample_thread.joinable())
            {
                sample_thread.join();
            }
        }

        void TeardownPipeline()
        {
            if(this->_pipeline != NULL)
            {
                GstState state, pending;
                gst_element_get_state(
                    GST_ELEMENT(this->_pipeline),
                    &state,
                    &pending,
                    GST_CLOCK_TIME_NONE);

                if(state != GST_STATE_NULL)
                {
                    gst_element_set_state(
                        GST_ELEMENT(this->_pipeline), GST_STATE_NULL);
                }

                gst_object_unref(this->_pipeline);
                this->_pipeline = NULL;
            }
        }
    };
}

class FeatureExtractorTestFixture
    : public ::testing::TestWithParam<GstCudaOfAlgorithm>
{
    protected:
    GstCudaOfAlgorithm algorithm_type;

    public:

    protected:
    void SetUp() override
    {
        this->algorithm_type = this->GetParam();
    }

    cv::Size_<size_t> CalculateBlockDimensions(
        cv::Size_<size_t> frame_dimensions,
        cv::Size_<size_t> grid_dimensions)
    {
        return cv::Size_<size_t>(
            ceil_div_int<size_t>(frame_dimensions.width, grid_dimensions.width),
            ceil_div_int<size_t>(
                frame_dimensions.height, grid_dimensions.height));
    }

    std::vector<float> ExtractFeatures(
        const cv::Mat optical_flow_matrix,
        cv::Size_<size_t> frame_dimensions,
        size_t optical_flow_vector_grid_size,
        float optical_flow_vector_threshold
        = default_magnitude_quadrant_threshold_squared,
        cv::Size_<size_t> feature_grid_dimensions
        = default_features_matrix_size)
    {
        std::vector<std::vector<float>> features_grid;

        cv::Size_<size_t> block_dimensions = this->CalculateBlockDimensions(
            frame_dimensions, feature_grid_dimensions);

        cv::Point_<size_t> block_index;

        for(block_index.x = 0; block_index.x < feature_grid_dimensions.width;
            block_index.x++)
        {
            auto features_row = std::vector<float>();
            for(block_index.y = 0;
                block_index.y < feature_grid_dimensions.height;
                block_index.y++)
            {
                features_row.push_back(this->ExtractFeaturesForBlock(
                    block_index,
                    block_dimensions,
                    optical_flow_matrix,
                    frame_dimensions,
                    optical_flow_vector_grid_size,
                    optical_flow_vector_threshold));
            }

            features_grid.push_back(features_row);
        }

        std::vector<float> features_array;

        for(std::size_t y = 0; y < feature_grid_dimensions.height; y++)
        {
            for(std::size_t x = 0; x < feature_grid_dimensions.width; x++)
            {
                features_array.push_back(features_grid[x][y]);
            }
        }

        std::size_t aggregations_array_size = ceil_div_int<size_t>(
            features_array.size(), features_per_aggregation);

        std::vector<float> aggregations_array(aggregations_array_size);

        for(size_t aggregate_idx = 0; aggregate_idx < aggregations_array_size;
            aggregate_idx++)
        {
            float maximum_spatial_magnitude = 0.0f;

            for(size_t idx = 0;
                idx < features_per_aggregation
                && (aggregate_idx * features_per_aggregation + idx)
                       < features_array.size();
                idx++)
            {
                maximum_spatial_magnitude = std::max(
                    maximum_spatial_magnitude,
                    features_array
                        [aggregate_idx * features_per_aggregation + idx]);
            }

            aggregations_array[aggregate_idx] = maximum_spatial_magnitude;
        }

        return aggregations_array;
    }

    float ExtractFeaturesForBlock(
        cv::Point_<size_t> block_index,
        cv::Size_<size_t> block_dimensions,
        const cv::Mat optical_flow_matrix,
        cv::Size_<size_t> frame_dimensions,
        size_t optical_flow_vector_grid_size,
        float flow_vector_threshold)
    {
        float spatial_magnitude = 0.0f;
        cv::Point_<size_t> thread_index;

        for(thread_index.x = 0; thread_index.x < block_dimensions.width;
            thread_index.x++)
        {
            for(thread_index.y = 0; thread_index.y < block_dimensions.height;
                thread_index.y++)
            {
                cv::Point_<size_t> frame_index(
                    block_index.x * block_dimensions.width + thread_index.x,
                    block_index.y * block_dimensions.height + thread_index.y);
                cv::Point_<size_t> optical_flow_index(
                    frame_index.x / optical_flow_vector_grid_size,
                    frame_index.y / optical_flow_vector_grid_size);

                if(frame_index.y < frame_dimensions.height
                   && frame_index.x < frame_dimensions.width
                   && optical_flow_index.y < static_cast<const unsigned int>(
                          optical_flow_matrix.rows)
                   && optical_flow_index.x < static_cast<const unsigned int>(
                          optical_flow_matrix.cols))
                {
                    cv::Vec2s flow_vector
                        = optical_flow_matrix.at<cv::Vec2s>(optical_flow_index);

                    float flow_vector_x = static_cast<float>(
                        flow_vector[0] / static_cast<float>(1 << 5));
                    float flow_vector_y = static_cast<float>(
                        flow_vector[1] / static_cast<float>(1 << 5));

                    float flow_vector_x_squared = flow_vector_x * flow_vector_x;
                    float flow_vector_y_squared = flow_vector_y * flow_vector_y;

                    if(flow_vector_x_squared > flow_vector_threshold)
                    {
                        spatial_magnitude += std::abs(flow_vector_x);
                    }

                    if(flow_vector_y_squared > flow_vector_threshold)
                    {
                        spatial_magnitude += std::abs(flow_vector_y);
                    }
                }
            }
        }

        return spatial_magnitude;
    }
};

TEST_P(FeatureExtractorTestFixture, TestFeatureExtractor)
{
    auto sample_queue = std::make_shared<std::queue<GstSample *>>();

    GEnumClass *klass
        = G_ENUM_CLASS(g_type_class_ref(GST_TYPE_CUDA_OF_ALGORITHM));

    switch(this->algorithm_type)
    {
        case OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0:
            ASSERT_NE(g_enum_get_value_by_nick(klass, "nvidia-1.0"), nullptr);
            break;
        case OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0:
            ASSERT_NE(g_enum_get_value_by_nick(klass, "nvidia-2.0"), nullptr);
            break;
        case OPTICAL_FLOW_ALGORITHM_FARNEBACK:
            ASSERT_NE(g_enum_get_value_by_nick(klass, "farneback"), nullptr);
            break;
        default:
            throw std::logic_error(
                "We somehow received an algorithm type that isn't "
                "supported or should not be tested here.");
            break;
    }

    {
        TestFeatureExtractorPipeline pipeline(
            test_frame_size.width,
            test_frame_size.height,
            test_framerate,
            this->algorithm_type,
            sample_queue);
        pipeline.Run();
    }

    {
        bool is_first_frame = true;
        EXPECT_EQ(sample_queue->size(), 2u);

        while(!sample_queue->empty())
        {
            GstSample *sample = sample_queue->front();

            EXPECT_NE(sample, nullptr);

            if(sample != NULL)
            {
                GstBuffer *buffer = gst_sample_get_buffer(sample);

                /*
                     * The first frame won't have any optical flow or
                     * feature-extractor metadata, so there's no point checking
                     * it.
                     *
                     * - J.O.
                     */
                if(!is_first_frame)
                {
                    GstMetaOpticalFlow *optical_flow_metadata
                        = GST_META_OPTICAL_FLOW_GET(buffer);
                    EXPECT_NE(optical_flow_metadata, nullptr);

                    GstMetaAlgorithmFeatures *feature_extractor_metadata
                        = GST_META_ALGORITHM_FEATURES_GET(buffer);
                    EXPECT_NE(feature_extractor_metadata, nullptr);
                    EXPECT_NE(feature_extractor_metadata->features, nullptr);

                    if((this->algorithm_type
                            == OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0
                        || this->algorithm_type
                               == OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0)
                       && optical_flow_metadata != nullptr
                       && optical_flow_metadata->optical_flow_vectors != nullptr
                       && feature_extractor_metadata != nullptr)
                    {
                        cv::Mat host_optical_flow_matrix;
                        optical_flow_metadata->optical_flow_vectors->download(
                            host_optical_flow_matrix);

                        // We're using the default feature-grid dimensions
                        // and motion thresholds for now within the test
                        // pipeline. So leave the CPU feature-extractor
                        // with those as the defaults as well.
                        auto aggregate_features_array = this->ExtractFeatures(
                            host_optical_flow_matrix,
                            test_frame_size,
                            optical_flow_metadata
                                ->optical_flow_vector_grid_size);

                        std::size_t expected_array_size = ceil_div_int<size_t>(
                            default_features_matrix_size.area(),
                            features_per_aggregation);

                        gsize features_array_length
                            = feature_extractor_metadata->features->len;

                        EXPECT_EQ(
                            aggregate_features_array.size(),
                            expected_array_size);
                        EXPECT_EQ(features_array_length, expected_array_size);

                        for(size_t idx = 0; idx < expected_array_size; idx++)
                        {
                            auto host_spatial_magnitude
                                = aggregate_features_array[idx];
                            gfloat gpu_spatial_magnitude = g_array_index(
                                feature_extractor_metadata->features,
                                gfloat,
                                idx);

                            EXPECT_NEAR(
                                host_spatial_magnitude,
                                gpu_spatial_magnitude,
                                10.0f);
                        }
                    }
                }

                gst_sample_unref(sample);
            }

            sample_queue->pop();
            is_first_frame = false;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    FeatureExtractorTests,
    FeatureExtractorTestFixture,
    Values(
        OPTICAL_FLOW_ALGORITHM_FARNEBACK,
        OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0,
        OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0));
