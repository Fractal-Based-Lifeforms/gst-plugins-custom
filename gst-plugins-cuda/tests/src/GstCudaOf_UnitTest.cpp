#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <thread>

#include <Poco/Path.h>
#include <gst/app/gstappsink.h>
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

#include "gstcudaof.h"

using ::testing::Values;

namespace
{
    const Poco::Path default_frames_path
        = Poco::Path(std::string(ROOT_DATA_DIRECTORY) + std::string("/frames/"))
              .absolute();

    class TestOpticalFlowPipeline
    {
        private:
        GstCudaOfAlgorithm _algorithm_type = OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0;
        guint _bus_watch_id = 0;
        std::size_t _frame_width = 0;
        std::size_t _frame_height = 0;
        std::uint32_t _framerate = 0;
        GMainLoop *_loop = nullptr;
        GstPipeline *_pipeline = nullptr;
        std::shared_ptr<std::queue<GstSample *>> _sample_queue = nullptr;

        public:
        TestOpticalFlowPipeline(
            const std::size_t frame_width,
            const std::size_t frame_height,
            const std::uint32_t framerate,
            const GstCudaOfAlgorithm algorithm_type,
            const std::shared_ptr<std::queue<GstSample *>> sample_queue)
        {
            this->_bus_watch_id = 0;
            this->_loop = nullptr;
            this->_pipeline = nullptr;

            this->_frame_width = frame_width;
            this->_frame_height = frame_height;
            this->_framerate = framerate;
            this->_algorithm_type = algorithm_type;

            this->_sample_queue = sample_queue;

            this->CreatePipeline();
        }
        ~TestOpticalFlowPipeline()
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
            auto *that = static_cast<TestOpticalFlowPipeline *>(user_data);

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
                    TestOpticalFlowPipeline::MonitorMessageBus),
                static_cast<gpointer>(this));

            gst_element_set_state(
                GST_ELEMENT(this->_pipeline), GST_STATE_PLAYING);

            std::thread sample_thread
                = std::thread(&TestOpticalFlowPipeline::SampleThread, this);

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

    class DenseOpticalFlowTestFixture
        : public ::testing::TestWithParam<GstCudaOfAlgorithm>
    {
        protected:
        cv::Ptr<cv::cuda::DenseOpticalFlow> algorithm = nullptr;
        GstCudaOfAlgorithm algorithm_type;

        public:
        cv::cuda::GpuMat ExtractFrameFromFile(
            std::ifstream &frame_file,
            const std::size_t frame_width,
            const std::size_t frame_height,
            const std::uint8_t num_of_channels)
        {
            cv::cuda::GpuMat frame_gpu_mat;
            std::size_t file_size = 0;

            frame_file.seekg(0, std::ios::end);
            file_size = frame_file.tellg();
            frame_file.seekg(0, std::ios::beg);

            std::vector<std::uint8_t> frame_data(file_size);
            frame_file.read(
                reinterpret_cast<char *>(frame_data.data()), file_size);

            cv::Mat frame_mat = cv::Mat(
                frame_height,
                frame_width,
                CV_MAKETYPE(CV_8U, num_of_channels),
                frame_data.data());

            frame_gpu_mat.upload(frame_mat);

            return frame_gpu_mat;
        }

        cv::cuda::GpuMat CalculateDenseOpticalFlow(
            std::ifstream &first_frame_file,
            std::ifstream &second_frame_file,
            const std::size_t frame_width,
            const std::size_t frame_height,
            const std::uint8_t num_of_channels)
        {
            cv::cuda::GpuMat output_vectors;

            cv::cuda::GpuMat first_frame_gpu_mat = this->ExtractFrameFromFile(
                first_frame_file, frame_width, frame_height, num_of_channels);
            cv::cuda::GpuMat second_frame_gpu_mat = this->ExtractFrameFromFile(
                second_frame_file, frame_width, frame_height, num_of_channels);
            this->algorithm->calc(
                first_frame_gpu_mat, second_frame_gpu_mat, output_vectors);

            return output_vectors;
        }

        protected:
        void SetUp() override
        {
            this->algorithm_type = this->GetParam();

            switch(this->algorithm_type)
            {
                case OPTICAL_FLOW_ALGORITHM_FARNEBACK:
                    this->algorithm = cv::cuda::FarnebackOpticalFlow::create();
                    break;
                default:
                    break;
            }
        }
    };

    TEST_P(DenseOpticalFlowTestFixture, TestDenseOpticalFlow)
    {
        cv::Mat optical_flow_vectors;
        auto sample_queue = std::make_shared<std::queue<GstSample *>>();
        /*
         * First, we need to calculate the optical flow vectors with OpenCV
         * directly. This is to determine the output that we should be
         * expecting from the plugin.
         *
         * - J.O.
         */
        {
            GEnumClass *klass
                = G_ENUM_CLASS(g_type_class_ref(GST_TYPE_CUDA_OF_ALGORITHM));

            switch(this->algorithm_type)
            {
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

            g_type_class_unref(klass);

            Poco::Path first_frame_file_path
                = Poco::Path(default_frames_path, "sample_1080p_h264.0001.raw")
                      .absolute();
            ASSERT_TRUE(first_frame_file_path.isFile());

            Poco::Path second_frame_file_path
                = Poco::Path(default_frames_path, "sample_1080p_h264.0002.raw")
                      .absolute();
            ASSERT_TRUE(second_frame_file_path.isFile());

            std::ifstream first_frame_file(
                first_frame_file_path.toString(),
                std::ios::in | std::ios::binary);
            ASSERT_TRUE(first_frame_file.is_open());

            std::ifstream second_frame_file(
                second_frame_file_path.toString(),
                std::ios::in | std::ios::binary);
            ASSERT_TRUE(second_frame_file.is_open());

            this->CalculateDenseOpticalFlow(
                    first_frame_file, second_frame_file, 1920, 1080, 1)
                .download(optical_flow_vectors);
        }

        /*
         * Then, we run a pipeline with the GstCudaOf plugin/element in order to
         * allow the pipeline to calculate the optical flow data.
         *
         * - J.O.
         */

        {
            TestOpticalFlowPipeline pipeline(
                1920, 1080, 5, this->algorithm_type, sample_queue);
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
                     * The first frame won't have any optical flow metadata, so
                     * there's no point checking it.
                     *
                     * - J.O.
                     */
                    if(!is_first_frame)
                    {
                        GstMetaOpticalFlow *optical_flow_metadata
                            = GST_META_OPTICAL_FLOW_GET(buffer);
                        EXPECT_NE(optical_flow_metadata, nullptr);

                        if(optical_flow_metadata != NULL)
                        {
                            /*
                             * We can't check the optical flow vectors whilst
                             * they are in GPU memory. So we need to download
                             * them to the host.
                             *
                             * - J.O.
                             */
                            cv::Mat gstcuda_optical_flow_vectors;
                            optical_flow_metadata->optical_flow_vectors
                                ->download(gstcuda_optical_flow_vectors);

                            /*
                             * Then, as the matrix size should be the same for
                             * both, iterate over the GstCudaOf output matrix
                             * and confirm that both matricies contain equal
                             * values.
                             *
                             * - J.O.
                             */
                            EXPECT_EQ(
                                optical_flow_vectors.rows,
                                gstcuda_optical_flow_vectors.rows);
                            EXPECT_EQ(
                                optical_flow_vectors.cols,
                                gstcuda_optical_flow_vectors.cols);

                            for(int ii = 0;
                                ii < gstcuda_optical_flow_vectors.rows;
                                ii++)
                            {
                                for(int jj = 0;
                                    jj < gstcuda_optical_flow_vectors.cols;
                                    jj++)
                                {

                                    cv::Vec2f optical_flow_vector
                                        = optical_flow_vectors.at<cv::Vec2f>(
                                            ii, jj);
                                    cv::Vec2f gstcuda_optical_flow_vector
                                        = gstcuda_optical_flow_vectors
                                              .at<cv::Vec2f>(ii, jj);

                                    EXPECT_FLOAT_EQ(
                                        optical_flow_vector[0],
                                        gstcuda_optical_flow_vector[0]);
                                    EXPECT_FLOAT_EQ(
                                        optical_flow_vector[1],
                                        gstcuda_optical_flow_vector[1]);
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
        DenseOpticalFlowTests,
        DenseOpticalFlowTestFixture,
        Values(OPTICAL_FLOW_ALGORITHM_FARNEBACK));

    class NVIDIAOpticalFlowTestFixture
        : public ::testing::TestWithParam<GstCudaOfAlgorithm>
    {
        protected:
        cv::Ptr<cv::cuda::NvidiaHWOpticalFlow> algorithm = nullptr;
        GstCudaOfAlgorithm algorithm_type;

        public:
        cv::cuda::GpuMat ExtractFrameFromFile(
            std::ifstream &frame_file,
            const std::size_t frame_width,
            const std::size_t frame_height,
            const std::uint8_t num_of_channels)
        {
            cv::cuda::GpuMat frame_gpu_mat;
            std::size_t file_size = 0;

            frame_file.seekg(0, std::ios::end);
            file_size = frame_file.tellg();
            frame_file.seekg(0, std::ios::beg);

            std::vector<std::uint8_t> frame_data(file_size);
            frame_file.read(
                reinterpret_cast<char *>(frame_data.data()), file_size);

            cv::Mat frame_mat = cv::Mat(
                frame_height,
                frame_width,
                CV_MAKETYPE(CV_8U, num_of_channels),
                frame_data.data());

            frame_gpu_mat.upload(frame_mat);

            return frame_gpu_mat;
        }

        cv::cuda::GpuMat CalculateNVIDIAOpticalFlow(
            std::ifstream &first_frame_file,
            std::ifstream &second_frame_file,
            const std::size_t &frame_width,
            const std::size_t &frame_height,
            const std::uint8_t &num_of_channels)
        {
            cv::cuda::GpuMat output_vectors;

            cv::cuda::GpuMat first_frame_gpu_mat = this->ExtractFrameFromFile(
                first_frame_file, frame_width, frame_height, num_of_channels);
            cv::cuda::GpuMat second_frame_gpu_mat = this->ExtractFrameFromFile(
                second_frame_file, frame_width, frame_height, num_of_channels);
            this->algorithm->calc(
                first_frame_gpu_mat, second_frame_gpu_mat, output_vectors);

            return output_vectors;
        }

        protected:
        void SetUp() override
        {
            this->algorithm_type = this->GetParam();

            switch(this->algorithm_type)
            {
                case OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0:
                    this->algorithm = cv::cuda::NvidiaOpticalFlow_1_0::create(
                        cv::Size(1920, 1080),
                        cv::cuda::NvidiaOpticalFlow_1_0::NV_OF_PERF_LEVEL_FAST,
                        true,
                        false,
                        false,
                        0);
                    break;
                case OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0:
                    this->algorithm = cv::cuda::NvidiaOpticalFlow_2_0::create(
                        cv::Size(1920, 1080),
                        cv::cuda::NvidiaOpticalFlow_2_0::NV_OF_PERF_LEVEL_FAST,
                        cv::cuda::NvidiaOpticalFlow_2_0::
                            NV_OF_OUTPUT_VECTOR_GRID_SIZE_4,
                        cv::cuda::NvidiaOpticalFlow_2_0::
                            NV_OF_HINT_VECTOR_GRID_SIZE_4,
                        true,
                        false,
                        false,
                        0);
                    break;
                default:
                    break;
            }
        }
    };

    TEST_P(NVIDIAOpticalFlowTestFixture, TestNVIDIAOpticalFlow)
    {
        cv::Mat optical_flow_vectors;
        auto sample_queue = std::make_shared<std::queue<GstSample *>>();
        /*
         * First, we need to calculate the optical flow vectors with OpenCV
         * directly. This is to determine the output that we should be
         * expecting from the plugin.
         *
         * - J.O.
         */
        {
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
                default:
                    throw std::logic_error(
                        "We somehow received an algorithm type that isn't "
                        "supported or should not be tested here.");
                    break;
            }

            g_type_class_unref(klass);

            Poco::Path first_frame_file_path
                = Poco::Path(default_frames_path, "sample_1080p_h264.0001.raw")
                      .absolute();
            ASSERT_TRUE(first_frame_file_path.isFile());

            Poco::Path second_frame_file_path
                = Poco::Path(default_frames_path, "sample_1080p_h264.0002.raw")
                      .absolute();
            ASSERT_TRUE(second_frame_file_path.isFile());

            std::ifstream first_frame_file(
                first_frame_file_path.toString(),
                std::ios::in | std::ios::binary);
            ASSERT_TRUE(first_frame_file.is_open());

            std::ifstream second_frame_file(
                second_frame_file_path.toString(),
                std::ios::in | std::ios::binary);
            ASSERT_TRUE(second_frame_file.is_open());

            this->CalculateNVIDIAOpticalFlow(
                    first_frame_file, second_frame_file, 1920, 1080, 1)
                .download(optical_flow_vectors);
        }

        /*
         * Then, we run a pipeline with the GstCudaOf plugin/element in order to
         * allow the pipeline to calculate the optical flow data.
         *
         * - J.O.
         */

        {
            TestOpticalFlowPipeline pipeline(
                1920, 1080, 5, this->algorithm_type, sample_queue);
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
                     * The first frame won't have any optical flow metadata, so
                     * there's no point checking it.
                     *
                     * - J.O.
                     */
                    if(!is_first_frame)
                    {
                        GstMetaOpticalFlow *optical_flow_metadata
                            = GST_META_OPTICAL_FLOW_GET(buffer);
                        EXPECT_NE(optical_flow_metadata, nullptr);

                        if(optical_flow_metadata != NULL)
                        {
                            /*
                             * We can't check the optical flow vectors whilst
                             * they are in GPU memory. So we need to download
                             * them to the host.
                             *
                             * - J.O.
                             */
                            cv::Mat gstcuda_optical_flow_vectors;
                            optical_flow_metadata->optical_flow_vectors
                                ->download(gstcuda_optical_flow_vectors);

                            /*
                             * Then, as the matrix size should be the same for
                             * both, iterate over the GstCudaOf output matrix
                             * and confirm that both matricies contain equal
                             * values.
                             *
                             * - J.O.
                             */
                            EXPECT_EQ(
                                optical_flow_vectors.rows,
                                gstcuda_optical_flow_vectors.rows);
                            EXPECT_EQ(
                                optical_flow_vectors.cols,
                                gstcuda_optical_flow_vectors.cols);

                            for(int ii = 0;
                                ii < gstcuda_optical_flow_vectors.rows;
                                ii++)
                            {
                                for(int jj = 0;
                                    jj < gstcuda_optical_flow_vectors.cols;
                                    jj++)
                                {

                                    cv::Vec2s optical_flow_vector
                                        = optical_flow_vectors.at<cv::Vec2s>(
                                            ii, jj);
                                    cv::Vec2s gstcuda_optical_flow_vector
                                        = gstcuda_optical_flow_vectors
                                              .at<cv::Vec2s>(ii, jj);

                                    EXPECT_EQ(
                                        optical_flow_vector[0],
                                        gstcuda_optical_flow_vector[0]);
                                    EXPECT_EQ(
                                        optical_flow_vector[1],
                                        gstcuda_optical_flow_vector[1]);
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
        NVIDIAOpticalFlowTests,
        NVIDIAOpticalFlowTestFixture,
        Values(
            OPTICAL_FLOW_ALGORITHM_NVIDIA_1_0,
            OPTICAL_FLOW_ALGORITHM_NVIDIA_2_0));
}
