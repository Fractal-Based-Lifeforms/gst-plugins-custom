#include <cstdint>
#include <fstream>
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

#include "cudafeaturesmatrix.h"
#include "gstcudaof.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"

using ::testing::Values;

namespace
{
    const Poco::Path default_frames_path
        = Poco::Path(std::string(ROOT_DATA_DIRECTORY) + std::string("/frames/"))
              .absolute();
    const std::string kernel_source_location
        = Poco::Path(std::string(GST_CUDA_FEATURE_EXTRACTOR_KERNEL_SOURCE_PATH))
              .absolute()
              .toString();

    struct MotionFeatures
    {
        uint32_t pixels = 0u;
        uint32_t count = 0u;
        float x0_to_x1_magnitude = 0.0f;
        float x1_to_x0_magnitude = 0.0f;
        float y0_to_y1_magnitude = 0.0f;
        float y1_to_y0_magnitude = 0.0f;
    };

    struct MotionThresholds
    {
        float motion_threshold_squared = 4.0f;
        float magnitude_quadrant_threshold_squared = 2.25f;
    };

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

            if(GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS)
            {
                if(g_main_loop_is_running(that->_loop))
                {
                    g_main_loop_quit(that->_loop);
                }

                that->_bus_watch_id = 0;
                return FALSE;
            }
            else if(GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR)
            {
                GError *error = NULL;
                gchararray details = NULL;

                gst_message_parse_error(message, &error, &details);

                if(details != NULL)
                {
                    g_free(details);
                    details = NULL;
                }

                if(error != NULL)
                {
                    g_error_free(error);
                    error = NULL;
                }

                if(g_main_loop_is_running(that->_loop))
                {
                    g_main_loop_quit(that->_loop);
                }

                that->_bus_watch_id = 0;
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
                "framerate", GST_TYPE_FRACTION, this->_framerate, 1,
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
                "start-index", 1,
                "stop-index", 2,
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
                "format",  23,
                "framerate", this->_framerate, 1,
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
                "cuda-device-id", 0,
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
                "cuda-device-id", 0,
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

            if(this->_bus_watch_id != 0)
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
                ((frame_dimensions.width + grid_dimensions.width - 1)
                 / (grid_dimensions.width)),
                ((frame_dimensions.height + grid_dimensions.height - 1)
                 / (grid_dimensions.height)));
        }

        std::vector<std::vector<MotionFeatures>> ExtractFeatures(
            const cv::Mat optical_flow_matrix,
            cv::Size_<size_t> frame_dimensions,
            size_t optical_flow_vector_grid_size,
            MotionThresholds optical_flow_vector_thresholds = {4.0f, 2.25f},
            cv::Size_<size_t> feature_grid_dimensions
            = cv::Size_<size_t>(20, 20))
        {
            std::vector<std::vector<MotionFeatures>> features_grid;

            cv::Size_<size_t> block_dimensions = this->CalculateBlockDimensions(
                frame_dimensions, feature_grid_dimensions);

            cv::Point_<size_t> block_index;

            for(block_index.x = 0;
                block_index.x < feature_grid_dimensions.width;
                block_index.x++)
            {
                auto features_row = std::vector<MotionFeatures>();
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
                        optical_flow_vector_thresholds));
                }

                features_grid.push_back(features_row);
            }

            return features_grid;
        }

        MotionFeatures ExtractFeaturesForBlock(
            cv::Point_<size_t> block_index,
            cv::Size_<size_t> block_dimensions,
            const cv::Mat optical_flow_matrix,
            cv::Size_<size_t> frame_dimensions,
            size_t optical_flow_vector_grid_size,
            MotionThresholds flow_vector_thresholds)
        {
            MotionFeatures features;
            cv::Point_<size_t> thread_index;

            for(thread_index.x = 0; thread_index.x < block_dimensions.width;
                thread_index.x++)
            {
                for(thread_index.y = 0;
                    thread_index.y < block_dimensions.height;
                    thread_index.y++)
                {
                    cv::Point_<size_t> frame_index(
                        block_index.x * block_dimensions.width + thread_index.x,
                        block_index.y * block_dimensions.height
                            + thread_index.y);
                    cv::Point_<size_t> optical_flow_index(
                        frame_index.x / optical_flow_vector_grid_size,
                        frame_index.y / optical_flow_vector_grid_size);

                    if(frame_index.y < frame_dimensions.height
                       && frame_index.x < frame_dimensions.width
                       && optical_flow_index.y < optical_flow_matrix.rows
                       && optical_flow_index.x < optical_flow_matrix.cols)
                    {
                        cv::Vec2s flow_vector
                            = optical_flow_matrix.at<cv::Vec2s>(
                                optical_flow_index);

                        float flow_vector_x = static_cast<float>(
                            flow_vector[0] / static_cast<float>(1 << 5));
                        float flow_vector_y = static_cast<float>(
                            flow_vector[1] / static_cast<float>(1 << 5));

                        float flow_vector_x_squared
                            = flow_vector_x * flow_vector_x;
                        float flow_vector_y_squared
                            = flow_vector_y * flow_vector_y;
                        float distance_squared
                            = (flow_vector_x_squared + flow_vector_y_squared);

                        features.pixels++;

                        if(flow_vector_x_squared
                           > flow_vector_thresholds
                                 .magnitude_quadrant_threshold_squared)
                        {
                            if(flow_vector_x >= 0)
                            {
                                features.x0_to_x1_magnitude += flow_vector_x;
                            }
                            else
                            {
                                features.x1_to_x0_magnitude += -flow_vector_x;
                            }
                        }

                        if(flow_vector_y_squared
                           > flow_vector_thresholds
                                 .magnitude_quadrant_threshold_squared)
                        {
                            if(flow_vector_y >= 0)
                            {
                                features.y0_to_y1_magnitude += flow_vector_y;
                            }
                            else
                            {
                                features.y1_to_y0_magnitude += -flow_vector_y;
                            }
                        }

                        if(distance_squared
                           > flow_vector_thresholds.motion_threshold_squared)
                        {
                            features.count++;
                        }
                    }
                }
            }

            return features;
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
                ASSERT_FALSE(
                    g_enum_get_value_by_nick(klass, "nvidia-1.0") == NULL);
                break;
            case OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0:
                ASSERT_FALSE(
                    g_enum_get_value_by_nick(klass, "nvidia-2.0") == NULL);
                break;
            case OPTICAL_FLOW_ALGORITHM_FARNEBACK:
                ASSERT_FALSE(
                    g_enum_get_value_by_nick(klass, "farneback") == NULL);
                break;
            default:
                throw std::logic_error(
                    "We somehow received an algorithm type that isn't "
                    "supported or should not be tested here.");
                break;
        }

        {
            TestFeatureExtractorPipeline pipeline(
                1920, 1080, 5, this->algorithm_type, sample_queue);
            pipeline.Run();
        }

        {
            bool is_first_frame = true;
            EXPECT_EQ(sample_queue->size(), 2);

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
                        EXPECT_NE(
                            feature_extractor_metadata->features, nullptr);
                        EXPECT_TRUE(CUDA_IS_FEATURES_MATRIX(
                            feature_extractor_metadata->features));

                        if((this->algorithm_type
                                == OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0
                            || this->algorithm_type
                                   == OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0)
                           && optical_flow_metadata != nullptr
                           && optical_flow_metadata->optical_flow_vectors
                                  != nullptr
                           && feature_extractor_metadata != nullptr)
                        {
                            cv::Mat host_optical_flow_matrix;
                            optical_flow_metadata->optical_flow_vectors
                                ->download(host_optical_flow_matrix);

                            // We're using the default feature-grid dimensions
                            // and motion thresholds for now within the test
                            // pipeline. So leave the CPU feature-extractor
                            // with those as the defaults as well.
                            auto features_grid = this->ExtractFeatures(
                                host_optical_flow_matrix,
                                cv::Size_<size_t>(1920, 1080),
                                optical_flow_metadata
                                    ->optical_flow_vector_grid_size);

                            for(size_t ii = 0; ii < 20; ii++)
                            {
                                for(size_t jj = 0; jj < 20; jj++)
                                {
                                    auto features_cell = features_grid[jj][ii];
                                    auto cuda_features_cell
                                        = cuda_features_matrix_at(
                                            feature_extractor_metadata
                                                ->features,
                                            jj,
                                            ii);

                                    guint32 count;
                                    guint32 pixels;
                                    gfloat x0_to_x1_magnitude;
                                    gfloat x1_to_x0_magnitude;
                                    gfloat y0_to_y1_magnitude;
                                    gfloat y1_to_y0_magnitude;

                                    g_object_get(
                                        cuda_features_cell,
                                        "count",
                                        &count,
                                        "pixels",
                                        &pixels,
                                        "x0-to-x1-magnitude",
                                        &x0_to_x1_magnitude,
                                        "x1-to-x0-magnitude",
                                        &x1_to_x0_magnitude,
                                        "y0-to-y1-magnitude",
                                        &y0_to_y1_magnitude,
                                        "y1-to-y0-magnitude",
                                        &y1_to_y0_magnitude,
                                        nullptr);

                                    g_object_unref(cuda_features_cell);

                                    EXPECT_EQ(features_cell.count, count);
                                    EXPECT_EQ(features_cell.pixels, pixels);
                                    EXPECT_EQ(
                                        features_cell.x0_to_x1_magnitude,
                                        x0_to_x1_magnitude);
                                    EXPECT_EQ(
                                        features_cell.x1_to_x0_magnitude,
                                        x1_to_x0_magnitude);
                                    EXPECT_EQ(
                                        features_cell.y0_to_y1_magnitude,
                                        y0_to_y1_magnitude);
                                    EXPECT_EQ(
                                        features_cell.y1_to_y0_magnitude,
                                        y1_to_y0_magnitude);
                                }
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
}
