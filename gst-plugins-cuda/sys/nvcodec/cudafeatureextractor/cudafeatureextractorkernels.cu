typedef struct _CUDA2DPitchedArray
{
    void *device_ptr;
    size_t pitch;
    size_t width;
    size_t height;
    size_t elem_size;
} CUDA2DPitchedArray;

typedef struct _FrameDimensions
{
    size_t width;
    size_t height;
} FrameDimensions;

extern "C" __global__ void gst_cuda_feature_extractor_kernel(
    const CUDA2DPitchedArray flow_vector_matrix,
    const FrameDimensions frame_dimensions,
    const int flow_vector_grid_size,
    const float flow_vector_threshold,
    CUDA2DPitchedArray flow_features_matrix)
{
    unsigned int y_frame_idx = (((blockIdx.y * blockDim.y) + threadIdx.y));
    unsigned int x_frame_idx = (((blockIdx.x * blockDim.x) + threadIdx.x));
    unsigned int y_idx = (y_frame_idx / flow_vector_grid_size);
    unsigned int x_idx = (x_frame_idx / flow_vector_grid_size);

    __shared__ float block_spatial_magnitude;

    atomicExch(&block_spatial_magnitude, 0.0f);

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

        if(flow_vector_x_squared > flow_vector_threshold)
        {
            if(flow_vector_x >= 0)
            {
                atomicAdd((float *)&block_spatial_magnitude, flow_vector_x);
            }
            else
            {
                atomicAdd((float *)&block_spatial_magnitude, -flow_vector_x);
            }
        }

        if(flow_vector_y_squared > flow_vector_threshold)
        {
            if(flow_vector_y >= 0)
            {
                atomicAdd((float *)&block_spatial_magnitude, flow_vector_y);
            }
            else
            {
                atomicAdd((float *)&block_spatial_magnitude, -flow_vector_y);
            }
        }
    }

    __syncthreads();

    if(threadIdx.y + 1 == blockDim.y && threadIdx.x + 1 == blockDim.x)
    {
        unsigned int y_block_offset_index
            = blockIdx.y * flow_features_matrix.pitch;
        float *flow_spatial_feature
            = (float
                   *)((char *)(flow_features_matrix.device_ptr) + y_block_offset_index)
              + blockIdx.x;
        *flow_spatial_feature = block_spatial_magnitude;
    }
}

extern "C" __global__ void gst_cuda_feature_consolidation_kernel(
    const CUDA2DPitchedArray flow_spatial_feature_matrix,
    CUDA2DPitchedArray consolidated_flow_spatial_feature_matrix)
{
    unsigned int y_idx = ((blockIdx.y * blockDim.y) + threadIdx.y);
    unsigned int x_idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    __shared__ float consolidated_block_spatial_magnitude;

    atomicExch(&consolidated_block_spatial_magnitude, 0.0f);

    __syncthreads();

    if(y_idx < flow_spatial_feature_matrix.height
       && x_idx < (flow_spatial_feature_matrix.width / sizeof(float)))
    {
        unsigned int y_offset_index = y_idx * flow_spatial_feature_matrix.pitch;
        float *original_flow_spatial_feature
            = ((float
                    *)((char *)(flow_spatial_feature_matrix.device_ptr) + y_offset_index)
               + x_idx);

        atomicAdd(
            (float *)(&consolidated_block_spatial_magnitude),
            *original_flow_spatial_feature);
    }

    __syncthreads();

    if(threadIdx.y + 1 == blockDim.y && threadIdx.x + 1 == blockDim.x)
    {

        unsigned int y_block_offset_index
            = blockIdx.y * consolidated_flow_spatial_feature_matrix.pitch;
        float *flow_spatial_feature
            = (float
                   *)((char *)(consolidated_flow_spatial_feature_matrix.device_ptr) + y_block_offset_index)
              + blockIdx.x;
        *flow_spatial_feature = consolidated_block_spatial_magnitude;
    }
}
