// *****************************************************************************
// Filename:    device_graph_data.h
// Date:        2012-12-25 13:32
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef DEVICE_GRAPH_DATA_H_
#define DEVICE_GRAPH_DATA_H_

// NOTE: This file could only be involved in one and only one .cu file.

// Following are data structures residing on gpu. Each gpu has one copy of them.
// When we use cudaSetDevice(i) and then access these members, we get the copy
// from the ith gpu. We need to initialize them by copying the data from each
// thread's GPUControlThreadData.

__device__ GPUStatus d_gpu_status;

// TODO(laigd): If there are 'out' members defined by user in Global, we should
// define it as __device__ and we may need some mechanism to update those
// members since there are multiple gpus. For example, in max flow algorithm,
// there should be an merge function to merge the global 'max_flow' member.
__constant__ Global d_global;

__constant__ VertexContent d_vcon;
__constant__ EdgeContent d_econ;
__constant__ MessageContent d_mcon_recv;
__constant__ MessageContent d_mcon_send;
__constant__ AuxiliaryDeviceData d_auxiliary;

#endif
