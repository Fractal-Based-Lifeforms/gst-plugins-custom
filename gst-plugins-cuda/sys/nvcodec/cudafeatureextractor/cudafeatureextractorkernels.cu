typedef struct _MotionThresholds
{
    float motion_threshold_squared;
    float magnitude_quadrant_threshold_squared;
} MotionThresholds;

typedef struct _MotionFeatures
{
    unsigned int pixels;
    unsigned int count;
    float x0_to_x1_magnitude;
    float x1_to_x0_magnitude;
    float y0_to_y1_magnitude;
    float y1_to_y0_magnitude;
} MotionFeatures;

typedef struct _FrameDimensions
{
    size_t width;
    size_t height;
} FrameDimensions;

typedef struct _CUDA2DPitchedArray
{
    void *device_ptr;
    size_t pitch;
    size_t width;
    size_t height;
    size_t elem_size;
} CUDA2DPitchedArray;

extern "C" __global__ void gst_cuda_feature_extractor_kernel(
    const CUDA2DPitchedArray flow_vector_matrix,
    const FrameDimensions frame_dimensions,
    const int flow_vector_grid_size,
    const MotionThresholds flow_vector_thresholds,
    CUDA2DPitchedArray flow_features_matrix)
{
    unsigned int y_frame_idx = (((blockIdx.y * blockDim.y) + threadIdx.y));
    unsigned int x_frame_idx = (((blockIdx.x * blockDim.x) + threadIdx.x));
    unsigned int y_idx = (y_frame_idx / flow_vector_grid_size);
    unsigned int x_idx = (x_frame_idx / flow_vector_grid_size);

    __shared__ unsigned int block_count;
    __shared__ unsigned int block_pixels;
    __shared__ float block_x0_to_x1_magnitude;
    __shared__ float block_x1_to_x0_magnitude;
    __shared__ float block_y0_to_y1_magnitude;
    __shared__ float block_y1_to_y0_magnitude;

    atomicExch(&block_count, 0);
    atomicExch(&block_pixels, 0);
    atomicExch(&block_x0_to_x1_magnitude, 0.0f);
    atomicExch(&block_x1_to_x0_magnitude, 0.0f);
    atomicExch(&block_y0_to_y1_magnitude, 0.0f);
    atomicExch(&block_y1_to_y0_magnitude, 0.0f);

    __syncthreads();

    if(y_frame_idx < frame_dimensions.height
       && x_frame_idx < frame_dimensions.width
       && y_idx < flow_vector_matrix.height
       && x_idx < (flow_vector_matrix.width / flow_vector_matrix.elem_size))
    {
        unsigned int y_offset_index = y_idx * flow_vector_matrix.pitch;

        float flow_vector_x = 0.0;
        float flow_vector_y = 0.0;

        switch(flow_vector_matrix.elem_size)
        {
            case sizeof(short2):
                {
                    short2 *flow_vectors
                        = (short2
                               *)((char *)(flow_vector_matrix.device_ptr) + y_offset_index)
                          + x_idx;

                    flow_vector_x = (float)(flow_vectors->x / (float)(1 << 5));
                    flow_vector_y = (float)(flow_vectors->y / (float)(1 << 5));
                }
                break;
            case sizeof(float2):
                {
                    float2 *flow_vectors
                        = (float2
                               *)((char *)(flow_vector_matrix.device_ptr) + y_offset_index)
                          + x_idx;
                    flow_vector_x = flow_vectors->x;
                    flow_vector_y = flow_vectors->y;
                }
                break;
            default:
                break;
        }

        float flow_vector_x_squared = flow_vector_x * flow_vector_x;
        float flow_vector_y_squared = flow_vector_y * flow_vector_y;
        float distance_squared
            = (flow_vector_x_squared + flow_vector_y_squared);

        atomicAdd((unsigned int *)&block_pixels, 1);

        if(flow_vector_x_squared
           > flow_vector_thresholds.magnitude_quadrant_threshold_squared)
        {
            if(flow_vector_x >= 0)
            {
                atomicAdd((float *)&block_x0_to_x1_magnitude, flow_vector_x);
            }
            else
            {
                atomicAdd((float *)&block_x1_to_x0_magnitude, -flow_vector_x);
            }
        }

        if(flow_vector_y_squared
           > flow_vector_thresholds.magnitude_quadrant_threshold_squared)
        {
            if(flow_vector_y >= 0)
            {
                atomicAdd((float *)&block_y0_to_y1_magnitude, flow_vector_y);
            }
            else
            {
                atomicAdd((float *)&block_y1_to_y0_magnitude, -flow_vector_y);
            }
        }

        if(distance_squared > flow_vector_thresholds.motion_threshold_squared)
        {
            atomicAdd((unsigned int *)&block_count, 1);
        }
    }

    __syncthreads();

    if(threadIdx.y + 1 == blockDim.y && threadIdx.x + 1 == blockDim.x)
    {
        unsigned int y_block_offset_index
            = blockIdx.y * flow_features_matrix.pitch;
        MotionFeatures *flow_features
            = (MotionFeatures
                   *)((char *)(flow_features_matrix.device_ptr) + y_block_offset_index)
              + blockIdx.x;
        flow_features->pixels = block_pixels;
        flow_features->count = block_count;
        flow_features->x0_to_x1_magnitude = block_x0_to_x1_magnitude;
        flow_features->x1_to_x0_magnitude = block_x1_to_x0_magnitude;
        flow_features->y0_to_y1_magnitude = block_y0_to_y1_magnitude;
        flow_features->y1_to_y0_magnitude = block_y1_to_y0_magnitude;
    }
}

extern "C" __global__ void gst_cuda_feature_consolidation_kernel(
    const CUDA2DPitchedArray flow_features_matrix,
    CUDA2DPitchedArray consolidated_flow_features_matrix)
{
    unsigned int y_idx = ((blockIdx.y * blockDim.y) + threadIdx.y);
    unsigned int x_idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    __shared__ unsigned int consolidated_block_count;
    __shared__ unsigned int consolidated_block_pixels;
    __shared__ float consolidated_block_x0_to_x1_magnitude;
    __shared__ float consolidated_block_x1_to_x0_magnitude;
    __shared__ float consolidated_block_y0_to_y1_magnitude;
    __shared__ float consolidated_block_y1_to_y0_magnitude;

    atomicExch(&consolidated_block_count, 0);
    atomicExch(&consolidated_block_pixels, 0);
    atomicExch(&consolidated_block_x0_to_x1_magnitude, 0.0f);
    atomicExch(&consolidated_block_x1_to_x0_magnitude, 0.0f);
    atomicExch(&consolidated_block_y0_to_y1_magnitude, 0.0f);
    atomicExch(&consolidated_block_y1_to_y0_magnitude, 0.0f);

    __syncthreads();

    if(y_idx < flow_features_matrix.height
       && x_idx < (flow_features_matrix.width / sizeof(MotionFeatures)))
    {
        unsigned int y_offset_index = y_idx * flow_features_matrix.pitch;
        MotionFeatures *original_flow_features
            = ((MotionFeatures
                    *)((char *)(flow_features_matrix.device_ptr) + y_offset_index)
               + x_idx);

        atomicAdd(
            (unsigned int *)(&consolidated_block_pixels),
            original_flow_features->pixels);
        atomicAdd(
            (unsigned int *)(&consolidated_block_count),
            original_flow_features->count);
        atomicAdd(
            (float *)(&consolidated_block_x0_to_x1_magnitude),
            original_flow_features->x0_to_x1_magnitude);
        atomicAdd(
            (float *)(&consolidated_block_x1_to_x0_magnitude),
            original_flow_features->x1_to_x0_magnitude);
        atomicAdd(
            (float *)(&consolidated_block_y0_to_y1_magnitude),
            original_flow_features->y0_to_y1_magnitude);
        atomicAdd(
            (float *)(&consolidated_block_y1_to_y0_magnitude),
            original_flow_features->y1_to_y0_magnitude);
    }

    __syncthreads();

    if(threadIdx.y + 1 == blockDim.y && threadIdx.x + 1 == blockDim.x)
    {

        unsigned int y_block_offset_index
            = blockIdx.y * consolidated_flow_features_matrix.pitch;
        MotionFeatures *flow_features
            = (MotionFeatures
                   *)((char *)(consolidated_flow_features_matrix.device_ptr) + y_block_offset_index)
              + blockIdx.x;
        flow_features->pixels = consolidated_block_pixels;
        flow_features->count = consolidated_block_count;
        flow_features->x0_to_x1_magnitude
            = consolidated_block_x0_to_x1_magnitude;
        flow_features->x1_to_x0_magnitude
            = consolidated_block_x1_to_x0_magnitude;
        flow_features->y0_to_y1_magnitude
            = consolidated_block_y0_to_y1_magnitude;
        flow_features->y1_to_y0_magnitude
            = consolidated_block_y1_to_y0_magnitude;
    }
}
