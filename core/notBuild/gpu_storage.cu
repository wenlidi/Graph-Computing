// *****************************************************************************
// Filename:    gpu_storage.cc
// Date:        2012-12-25 10:01
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "gpu_storage.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "config.h"
#include "constants.h"
#include "device_graph_data.h"
#include "device_graph_data_types.h"
#include "device_hash_functions.h"
#include "device_util.h"
#include "edge_content_manager.h"
#include "gpu_status.h"
#include "gpu_storage.h"
#include "user_api.h"
#include "vertex_content_manager.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "---> [GPUStorageManager]: "
#endif


/********************* Helper data structure and functions ********************/

struct SortVconById_LT {
   __device__ bool operator()(const unsigned int idx1, const unsigned int idx2) {
    return d_vcon.d_id[idx1] < d_vcon.d_id[idx2];
  }
};

struct SortVconByInEdgeCount_LT {
  __device__ bool operator()(const unsigned int idx1, const unsigned int idx2) {
    return d_vcon.d_in_edge_count[idx1] == d_vcon.d_in_edge_count[idx2]
        ? d_vcon.d_out_edge_count[idx1] > d_vcon.d_out_edge_count[idx2]
        : d_vcon.d_in_edge_count[idx1] > d_vcon.d_in_edge_count[idx2];
  }
};

__global__ void K_SelfGather(
    const unsigned int *src,
    const unsigned int size,
    unsigned int *dst) {
  GET_NUM_THREAD_TID_AND_RETURN_IF_TID_GE(size);
  for (unsigned int i = tid; i < size; i += num_threads) {
    dst[src[i]] = i;
  }
}

__global__ void K_Invert(
    const unsigned int *src,
    const unsigned int size,
    unsigned int *dst) {
  GET_NUM_THREAD_TID_AND_RETURN_IF_TID_GE(size);
  for (unsigned int i = tid; i < size; i += num_threads) {
    dst[i] = src[size - 1 - i];
  }
}

// @value_to_find contains @num_value vertex IDs, and @vid_index[i] is the final
// position of vertex who owns id i. What we are going to do in this function is
// let out_index[i] to be the position of vertex who owns id value_to_find[i].
__global__ void K_SingleGPUFindSortedVidIndex(
    const unsigned int *vid_index,
    const unsigned int *value_to_find,
    const unsigned int num_value,
    unsigned int *out_index) {
  GET_NUM_THREAD_TID_AND_RETURN_IF_TID_GE(num_value);
  for (unsigned int i = tid; i < num_value; i += num_threads) {
    out_index[i] = vid_index[value_to_find[i]];
  }
}

// @array_of_blocks contains several blocks of data. Each block contains all
// same elements, the first block are all 0s, the second all 1s and so on, and
// if block A is presented earlier than block B in the array, then the size of
// block A is bigger than or equal to the size of block B.
//
// @prefix_sum_of_sorted_block_size contains the prefix sum of the sorted (from
// small to large) block size of each block in array_of_blocks.
__global__ void K_SingleGPUTranspose(
    const unsigned int *array_of_blocks,
    const unsigned int array_size,
    const unsigned int *prefix_sum_of_sorted_block_size,
    const unsigned int num_blocks,
    unsigned int *trans_index) {
  GET_NUM_THREAD_TID_AND_RETURN_IF_TID_GE(array_size);
  for (unsigned int i = tid; i < array_size; i += num_threads) {
    const unsigned int val = array_of_blocks[i];

    unsigned int pow;
    for (pow = 1; i >= pow && array_of_blocks[i - pow] == val; pow <<= 1);
    pow >>= 1;
    unsigned int first = i - pow;
    while (pow) {
      if (first >= pow && array_of_blocks[first - pow] == val) first -= pow;
      pow >>= 1;
    }
    unsigned int row = i - first;  // Starts from 0.

    unsigned int l = 0, r = num_blocks;
    while (l < r) {
      unsigned int mid = (l + r) >> 1;
      unsigned int cmp_val =
          mid == 0 ? prefix_sum_of_sorted_block_size[0] :
          prefix_sum_of_sorted_block_size[mid]
              - prefix_sum_of_sorted_block_size[mid - 1];
      if (row < cmp_val) {
        r = mid;
      } else {
        l = mid + 1;
      }
    }
    unsigned int result = (r == 0 ? 0 : prefix_sum_of_sorted_block_size[r - 1]);
    result += (num_blocks - r) * row;

    // We can add "array_of_blocks[i]" here because we assume that if k < i,
    // then the size of block which contains value of array_of_blocks[k] is
    // bigger than that of array_of_blocks[i].
    trans_index[i] = result + array_of_blocks[i];
  }
}

__global__ void K_SingleGPUCalculateInMsgNext(
    const unsigned int *array_of_blocks,
    const unsigned int *transpose_index,
    const unsigned int array_size,
    unsigned int *transposed_next) {
  GET_NUM_THREAD_TID_AND_RETURN_IF_TID_GE(array_size);
  for (unsigned int i = tid; i < array_size; i += num_threads) {
    if (i != array_size - 1 && array_of_blocks[i] == array_of_blocks[i + 1]) {
      transposed_next[transpose_index[i]] = transpose_index[i + 1];
    } else {
      transposed_next[transpose_index[i]] = ~0U;
    }
  }
}

__global__ void K_UserCompute() {
  const unsigned int num_vertex = d_vcon.d_size;
  GET_NUM_THREAD_TID_AND_RETURN_IF_TID_GE(num_vertex);

  for (unsigned int idx = tid; idx < num_vertex; idx += num_threads) {
    Vertex v(idx);
    MessageIterator msgs(idx);
    v.Compute(&msgs);
  }
}


/******************* GPUStorageManager function definition ********************/

GPUStorageManager::GPUStorageManager()
    : conf(NULL),
      msg_per_gpu_begin_calculated(false),
      msg_per_gpu_begin(NULL),
      msg_per_gpu_end(NULL) {
}

GPUStorageManager::~GPUStorageManager() {
  if (msg_per_gpu_begin != NULL) delete[] msg_per_gpu_begin;
  if (msg_per_gpu_end != NULL) delete[] msg_per_gpu_end;
}

void GPUStorageManager::Init(const Config *c) {
  conf = c;
  msg_per_gpu_begin = new unsigned int[conf->GetNumGPUControlThreads()];
  msg_per_gpu_end = new unsigned int[conf->GetNumGPUControlThreads()];
}

// Below are functions dealing with constant memory.

void GPUStorageManager::GetGPUStatusFromGPU(GPUStatus *gpu_status) {
  checkCudaErrors(cudaMemcpyFromSymbol(
        gpu_status, d_gpu_status, sizeof(*gpu_status), 0,
        cudaMemcpyDeviceToHost));
}

void GPUStorageManager::CopyGPUStatusToGPU(const GPUStatus &gpu_status) {
  checkCudaErrors(cudaMemcpyToSymbol(
        d_gpu_status, &gpu_status, sizeof(gpu_status), 0,
        cudaMemcpyHostToDevice));
}

void GPUStorageManager::CopyGlobalToGPU(
    const Global &global,
    cudaStream_t stream) {
  checkCudaErrors(cudaMemcpyToSymbolAsync(
        d_global, &global, sizeof(global), 0,
        cudaMemcpyHostToDevice, stream));
}

void GPUStorageManager::CopyVconToGPU(
    const VertexContent &vcon,
    cudaStream_t stream) {
  checkCudaErrors(cudaMemcpyToSymbolAsync(
        d_vcon, &vcon, sizeof(vcon), 0,
        cudaMemcpyHostToDevice, stream));
}

void GPUStorageManager::CopyEconToGPU(
    const EdgeContent &econ,
    cudaStream_t stream) {
  checkCudaErrors(cudaMemcpyToSymbolAsync(
        d_econ, &econ, sizeof(econ), 0,
        cudaMemcpyHostToDevice, stream));
}

void GPUStorageManager::CopyMconSendToGPU(
    const MessageContent &mcon_send,
    cudaStream_t stream) {
  checkCudaErrors(cudaMemcpyToSymbolAsync(
        d_mcon_send, &mcon_send, sizeof(mcon_send), 0,
        cudaMemcpyHostToDevice, stream));
}

void GPUStorageManager::CopyMconRecvToGPU(
    const MessageContent &mcon_recv,
    cudaStream_t stream) {
  checkCudaErrors(cudaMemcpyToSymbolAsync(
        d_mcon_recv, &mcon_recv, sizeof(mcon_recv), 0,
        cudaMemcpyHostToDevice, stream));
}

void GPUStorageManager::CopyAuxiliaryToGPU(
    const AuxiliaryDeviceData &auxiliary,
    cudaStream_t stream) {
  checkCudaErrors(cudaMemcpyToSymbolAsync(
        d_auxiliary, &auxiliary, sizeof(auxiliary), 0,
        cudaMemcpyHostToDevice, stream));
}

// Below are other public functions.

void GPUStorageManager::SingleGPUBuildIndexes(
    VertexContent *vcon,
    EdgeContent *econ,
    AuxiliaryDeviceData *auxiliary) {
  SingleGPUBuildVconIndexes(vcon);

#ifdef LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT
  // d_vid_index[i] is the final position of vertex who owns id i.
  unsigned int *d_vid_index = NULL;
  checkCudaErrors(cudaMalloc(&d_vid_index, vcon->d_size * sizeof(unsigned int)));
  K_SelfGather<<<GetNumBlocks(vcon->d_size), kDefaultNumThreadsPerBlock>>>(
      vcon->d_id, vcon->d_size, d_vid_index);

  SingleGPUBuildEconIndexes(d_vid_index, vcon->d_size, econ);
  // TODO(laigd): We may allocate @auxiliary here instead of doing so in
  // gpu_control_thread_data_types?
  SingleGPUBuildAuxiliaryIndexes(d_vid_index, *vcon, *econ, auxiliary);

  checkCudaErrors(cudaFree(d_vid_index));
#else
  SingleGPUBuildEconIndexes(NULL, vcon->d_size, econ);
  SingleGPUBuildAuxiliaryIndexes(*vcon, *econ, auxiliary);
#endif

  VertexContentManager::InitOutMembers(vcon);
  EdgeContentManager::InitOutMembers(econ);
}

void GPUStorageManager::UserCompute(VertexContent *vcon) {
  static unsigned int num_threads_per_block = conf->GetNumThreadsPerBlock();
  static unsigned int num_blocks =
      (vcon->d_size + num_threads_per_block - 1) / num_threads_per_block;
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "GPUStorageManager::UserCompute, "
       << "num_threads_per_block: " << num_threads_per_block
       << ", num_blocks: " << num_blocks
       << endl;
  );
#endif
  K_UserCompute<<<num_blocks, num_threads_per_block>>>();
  checkCudaErrors(cudaDeviceSynchronize());
}

// Below are private functions.

void GPUStorageManager::SingleGPUBuildVconIndexes(VertexContent *vcon) {
  unsigned int *d_shuffle_index, *d_tmp_buf;
  checkCudaErrors(cudaMalloc(&d_shuffle_index, vcon->d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_tmp_buf,       vcon->d_size * sizeof(unsigned int)));

  thrust::device_ptr<unsigned int> thr_shuffle_index(d_shuffle_index);
  thrust::sequence(thr_shuffle_index, thr_shuffle_index + vcon->d_size);
#ifdef LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT
  // Sort the vertex content according to in_edge_count.
  thrust::sort(thr_shuffle_index, thr_shuffle_index + vcon->d_size, SortVconByInEdgeCount_LT());
#else
  // Sort the vertex content according to its id.
  thrust::sort(thr_shuffle_index, thr_shuffle_index + vcon->d_size, SortVconById_LT());
#endif
  VertexContentManager::ShuffleInMembers(vcon, thr_shuffle_index, d_tmp_buf);

#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
  // Do nothing.
#else
  thrust::device_ptr<unsigned int> thr_in_edge_count(vcon->d_in_edge_count);
  thrust::inclusive_scan(thr_in_edge_count, thr_in_edge_count + vcon->d_size, thr_in_edge_count);
#endif

  // Do an inclusive scan on out_edge_count so that we can find the starting and
  // ending index of out edges of each vertex.
  thrust::device_ptr<unsigned int> thr_out_edge_count(vcon->d_out_edge_count);
  thrust::inclusive_scan(thr_out_edge_count, thr_out_edge_count + vcon->d_size, thr_out_edge_count);

  checkCudaErrors(cudaFree(d_shuffle_index));
  checkCudaErrors(cudaFree(d_tmp_buf));
}

void GPUStorageManager::SingleGPUBuildEconIndexes(
    const unsigned int *d_vid_index,
    const unsigned int num_vertexes,
    EdgeContent *econ) {
  unsigned int *d_edge_from_vid_index, *d_shuffle_index, *d_tmp_buf;
  checkCudaErrors(cudaMalloc(&d_edge_from_vid_index, econ->d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_shuffle_index,       econ->d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_tmp_buf,             econ->d_size * sizeof(unsigned int)));

  thrust::device_ptr<unsigned int> thr_rank(d_edge_from_vid_index);
#ifdef LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT
  K_SingleGPUFindSortedVidIndex<<<GetNumBlocks(econ->d_size), kDefaultNumThreadsPerBlock>>>(
      d_vid_index,
      econ->d_from,
      econ->d_size,
      d_edge_from_vid_index);
  checkCudaErrors(cudaDeviceSynchronize());
  // Now d_edge_from_vid_index stores the rank of each member of d_from.
#else
  thrust::device_ptr<unsigned int> thr_econ_from(econ->d_from);
  thrust::copy(thr_econ_from, thr_econ_from + econ->d_size, thr_rank);
#endif

  // Sort the edge content according to the rank.
  thrust::device_ptr<unsigned int> thr_shuffle_index(d_shuffle_index);
  thrust::sequence(thr_shuffle_index, thr_shuffle_index + econ->d_size);
  thrust::sort_by_key(thr_rank, thr_rank + econ->d_size, thr_shuffle_index);
  EdgeContentManager::ShuffleInMembers(econ, thr_shuffle_index, d_tmp_buf);

  checkCudaErrors(cudaFree(d_edge_from_vid_index));
  checkCudaErrors(cudaFree(d_shuffle_index));
  checkCudaErrors(cudaFree(d_tmp_buf));
}

#ifdef LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT

#ifdef LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS
void GPUStorageManager::SingleGPUBuildAuxiliaryIndexes(
    const unsigned int *d_vid_index,
    const VertexContent &vcon,
    const EdgeContent &econ,
    AuxiliaryDeviceData *auxiliary) {
#ifdef LAMBDA_DEBUG
  unsigned int *buf;
  checkCudaErrors(cudaMallocHost(&buf, std::max(vcon.d_size, econ.d_size) * sizeof(unsigned int)));
#endif
  unsigned int *d_edge_to_vid_index;
  unsigned int *d_in_edge_count_prefix_sum;
  unsigned int *d_org_shuffle_index;
  unsigned int *d_shuffle_index;
  unsigned int *d_transpose_index;
  checkCudaErrors(cudaMalloc(&d_edge_to_vid_index,        econ.d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_in_edge_count_prefix_sum, vcon.d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_org_shuffle_index,        econ.d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_shuffle_index,            econ.d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_transpose_index,          econ.d_size * sizeof(unsigned int)));

  K_SingleGPUFindSortedVidIndex<<<GetNumBlocks(econ.d_size), kDefaultNumThreadsPerBlock>>>(
      d_vid_index,
      econ.d_to,
      econ.d_size,
      d_edge_to_vid_index);
  checkCudaErrors(cudaDeviceSynchronize());

  thrust::device_ptr<unsigned int> thr_rank(d_edge_to_vid_index);
  thrust::device_ptr<unsigned int> thr_org_shuffle_index(d_org_shuffle_index);
  thrust::sequence(thr_org_shuffle_index, thr_org_shuffle_index + econ.d_size);
  thrust::sort_by_key(thr_rank, thr_rank + econ.d_size, thr_org_shuffle_index);

  K_Invert<<<GetNumBlocks(vcon.d_size), kDefaultNumThreadsPerBlock>>>(
      vcon.d_in_edge_count, vcon.d_size, d_in_edge_count_prefix_sum);
  thrust::device_ptr<unsigned int> thr_in_edge_count(d_in_edge_count_prefix_sum);
  thrust::inclusive_scan(thr_in_edge_count, thr_in_edge_count + vcon.d_size, thr_in_edge_count);

  K_SingleGPUTranspose<<<GetNumBlocks(econ.d_size), kDefaultNumThreadsPerBlock>>>(
      d_edge_to_vid_index,  // Must contain continuous blocks of natural number.
      econ.d_size,
      d_in_edge_count_prefix_sum,
      vcon.d_size,
      d_transpose_index);
  thrust::device_ptr<unsigned int> thr_shuffle_index(d_shuffle_index);
  thrust::device_ptr<unsigned int> thr_transpose_index(d_transpose_index);
  thrust::scatter(
      thr_org_shuffle_index, thr_org_shuffle_index + econ.d_size,
      thr_transpose_index, thr_shuffle_index);

  thrust::device_ptr<unsigned int> thr_from(econ.d_from);
  thrust::device_ptr<unsigned int> thr_in_msg_from(auxiliary->d_in_msg_from);
  thrust::gather(
      thr_shuffle_index, thr_shuffle_index + econ.d_size,
      thr_from, thr_in_msg_from);
  K_SingleGPUCalculateInMsgNext<<<GetNumBlocks(econ.d_size), kDefaultNumThreadsPerBlock>>>(
      d_edge_to_vid_index, d_transpose_index, econ.d_size, auxiliary->d_in_msg_next);
  K_SelfGather<<<GetNumBlocks(econ.d_size), kDefaultNumThreadsPerBlock>>>(
      d_shuffle_index, econ.d_size, auxiliary->d_out_edge_in_msg_map);

#ifdef LAMBDA_DEBUG
  DEBUG_OUTPUT(buf, d_vid_index,                "vid_index:                ", vcon.d_size, unsigned int);
  DEBUG_OUTPUT(buf, d_edge_to_vid_index,        "edge_to_vid_index:        ", econ.d_size, unsigned int);
  DEBUG_OUTPUT(buf, d_in_edge_count_prefix_sum, "in_edge_count_prefix_sum: ", vcon.d_size, unsigned int);
  DEBUG_OUTPUT(buf, d_org_shuffle_index,        "org_shuffle_index:        ", econ.d_size, unsigned int);
  DEBUG_OUTPUT(buf, d_shuffle_index,            "shuffle_index:            ", econ.d_size, unsigned int);
  DEBUG_OUTPUT(buf, d_transpose_index,          "transpose_index:          ", econ.d_size, unsigned int);
#endif
  checkCudaErrors(cudaFree(d_edge_to_vid_index));
  checkCudaErrors(cudaFree(d_in_edge_count_prefix_sum));
  checkCudaErrors(cudaFree(d_org_shuffle_index));
  checkCudaErrors(cudaFree(d_shuffle_index));
  checkCudaErrors(cudaFree(d_transpose_index));
#ifdef LAMBDA_DEBUG
  checkCudaErrors(cudaFreeHost(buf));
#endif
}
#else  // Sorting vertexes by in_edge_count without coalesced memory access.
void GPUStorageManager::SingleGPUBuildAuxiliaryIndexes(
    const unsigned int *d_vid_index,
    const VertexContent &vcon,
    const EdgeContent &econ,
    AuxiliaryDeviceData *auxiliary) {
  unsigned int *d_edge_to_vid_index, *d_shuffle_index;
  checkCudaErrors(cudaMalloc(&d_edge_to_vid_index, econ.d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_shuffle_index,     econ.d_size * sizeof(unsigned int)));

  K_SingleGPUFindSortedVidIndex<<<GetNumBlocks(econ.d_size), kDefaultNumThreadsPerBlock>>>(
      d_vid_index,
      econ.d_to,
      econ.d_size,
      d_edge_to_vid_index);
  checkCudaErrors(cudaDeviceSynchronize());

  thrust::device_ptr<unsigned int> thr_rank(d_edge_to_vid_index);
  thrust::device_ptr<unsigned int> thr_shuffle_index(d_shuffle_index);
  thrust::sequence(thr_shuffle_index, thr_shuffle_index + econ.d_size);
  thrust::sort_by_key(thr_rank, thr_rank + econ.d_size, thr_shuffle_index);

  thrust::device_ptr<unsigned int> thr_from(econ.d_from);
  thrust::device_ptr<unsigned int> thr_in_msg_from(auxiliary->d_in_msg_from);
  thrust::gather(
      thr_shuffle_index, thr_shuffle_index + econ.d_size,
      thr_from, thr_in_msg_from);
  K_SelfGather<<<GetNumBlocks(econ.d_size), kDefaultNumThreadsPerBlock>>>(
      d_shuffle_index, econ.d_size, auxiliary->d_out_edge_in_msg_map);

  checkCudaErrors(cudaFree(d_edge_to_vid_index));
  checkCudaErrors(cudaFree(d_shuffle_index));
}
#endif

#else  // Not sorting vertexes by in_edge_count
void GPUStorageManager::SingleGPUBuildAuxiliaryIndexes(
    const VertexContent &vcon,
    const EdgeContent &econ,
    AuxiliaryDeviceData *auxiliary) {
#ifdef LAMBDA_DEBUG
  unsigned int *buf;
  checkCudaErrors(cudaMallocHost(&buf, std::max(vcon.d_size, econ.d_size) * sizeof(unsigned int)));
#endif
  unsigned int *d_edge_to_vid_index, *d_shuffle_index;
  checkCudaErrors(cudaMalloc(&d_edge_to_vid_index, econ.d_size * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_shuffle_index,     econ.d_size * sizeof(unsigned int)));

  thrust::device_ptr<unsigned int> thr_rank(d_edge_to_vid_index);
  thrust::device_ptr<unsigned int> thr_econ_to(econ.d_to);
  thrust::copy(thr_econ_to, thr_econ_to + econ.d_size, thr_rank);

  thrust::device_ptr<unsigned int> thr_shuffle_index(d_shuffle_index);
  thrust::sequence(thr_shuffle_index, thr_shuffle_index + econ.d_size);
  thrust::sort_by_key(thr_rank, thr_rank + econ.d_size, thr_shuffle_index);

  thrust::device_ptr<unsigned int> thr_from(econ.d_from);
  thrust::device_ptr<unsigned int> thr_in_msg_from(auxiliary->d_in_msg_from);
  thrust::gather(
      thr_shuffle_index, thr_shuffle_index + econ.d_size,
      thr_from, thr_in_msg_from);
  K_SelfGather<<<GetNumBlocks(econ.d_size), kDefaultNumThreadsPerBlock>>>(
      d_shuffle_index, econ.d_size, auxiliary->d_out_edge_in_msg_map);

#ifdef LAMBDA_DEBUG
  DEBUG_OUTPUT(buf, d_edge_to_vid_index,              "edge_to_vid_index:   ", econ.d_size, unsigned int);
  DEBUG_OUTPUT(buf, d_shuffle_index,                  "shuffle_index:       ", econ.d_size, unsigned int);
  DEBUG_OUTPUT(buf, auxiliary->d_in_msg_from,         "in_msg_from:         ", econ.d_size, unsigned int);
  DEBUG_OUTPUT(buf, auxiliary->d_out_edge_in_msg_map, "out_edge_in_msg_map: ", econ.d_size, unsigned int);
#endif
  checkCudaErrors(cudaFree(d_edge_to_vid_index));
  checkCudaErrors(cudaFree(d_shuffle_index));
#ifdef LAMBDA_DEBUG
  checkCudaErrors(cudaFreeHost(buf));
#endif
}
#endif

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
