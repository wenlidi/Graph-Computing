// *****************************************************************************
// Filename:    gpu_control_thread_data_types.cc
// Date:        2012-12-09 22:51
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "gpu_control_thread_data_types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "auxiliary_manager.h"
#include "console_test_writer.h"
#include "console_writer.h"
#include "device_graph_data_types.h"
#include "dummy_writer.h"
#include "edge_content_manager.h"
#include "file_test_writer.h"
#include "global_manager.h"
#include "gpu_storage.h"
#include "host_hash_functions.h"
#include "message_content_manager.h"
#include "multiple_file_writer.h"
#include "single_file_writer.h"
#include "util.h"
#include "vertex_content_manager.h"

using std::cerr;
using std::cin;
using std::cout;
using std::endl;

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "---> [GPUControlThread " << control_thread_id << "]: "
#endif

#ifdef LAMBDA_PROFILING
#include "singleton.h"
#include "profiler.h"
#endif

GPUControlThreadData::GPUControlThreadData()
    : control_thread_id(-1),
      cuda_device_id(-1),
      // Controling data
      conf(NULL),
      shared_data(NULL),
      // Device data
      global(),
      vcon(),
      econ(),
      mcon_recv(),
      mcon_send(),
      auxiliary(),
      // Other member
      vcon_copied_size(0),
      econ_copied_size(0),
      local_gpu_status(),
      gpu_storage_manager(),
      writer(NULL) {
}

GPUControlThreadData::~GPUControlThreadData() {
  if (writer != NULL) delete writer;
}

void GPUControlThreadData::Init(
    const int id,
    const int device_id,
    const Config *config,
    SharedData *shared) {
  control_thread_id = id;
  cuda_device_id = device_id;
  conf = config;
  shared_data = shared;
  gpu_storage_manager.Init(conf);

  if (conf->GetWriterType() == WriterType::kConsoleWriter) {
    writer = new ConsoleWriter(&global, &vcon, &econ);
  } else if (conf->GetWriterType() == WriterType::kDummyWriter) {
    writer = new DummyWriter();
  } else if (conf->GetWriterType() == WriterType::kMultipleFileWriter) {
    writer = new MultipleFileWriter(
        conf, control_thread_id, &global, &vcon, &econ);
  } else if (conf->GetWriterType() == WriterType::kSingleFileWriter) {
    writer = new SingleFileWriter(conf, &global, &vcon, &econ);
  } else if (conf->GetWriterType() == WriterType::kConsoleTestWriter) {
    writer = new ConsoleTestWriter(&global, &vcon, &econ, shared_data);
  } else if (conf->GetWriterType() == WriterType::kFileTestWriter) {
    writer = new FileTestWriter(conf, &global, &vcon, &econ, shared_data);
  }
}

void GPUControlThreadData::Run() {
  checkCudaErrors(cudaSetDevice(cuda_device_id));
  cudaStream_t copy_stream;
  checkCudaErrors(cudaStreamCreate(&copy_stream));

  // Wait for reading of 'global'
  shared_data->WaitForGlobalRead();
  GlobalManager::Set(shared_data->GetGlobal(), &global);

  // Malloc vcon according to global
  VertexContentManager::Allocate(
      kHashGetNumVertexForGPU[conf->GetHashType()](
        global.d_num_vertex,
        conf->GetNumGPUControlThreads(),
        control_thread_id), &vcon);

  // Signal reading threads that vcon is ready to use
  shared_data->SignalVconMemoryReady();

  // Copy global and vcon to gpu
  gpu_storage_manager.CopyGlobalToGPU(global, copy_stream);
  gpu_storage_manager.CopyVconToGPU(vcon, copy_stream);

  // Wait for reading and copying of each 'in' member of 'vcon'
  WaitAndCopyVconToGPU();

  // Malloc econ according to read vcon data.
  EdgeContentManager::Allocate(
      shared_data->GetNumOutEdgesForGPU(control_thread_id), &econ);

  // Signal reading threads that econ is ready to use.
  shared_data->SignalEconMemoryReady();

  // Copy econ to gpu.
  gpu_storage_manager.CopyEconToGPU(econ, copy_stream);

  // Malloc mcon_recv and mcon_send on device and copy it to device memory
  MessageContentManager::Allocate(
      shared_data->GetNumInEdgesForGPU(control_thread_id), &mcon_recv);
  gpu_storage_manager.CopyMconRecvToGPU(mcon_recv, copy_stream);

  MessageContentManager::Allocate(econ.d_size, &mcon_send);
  gpu_storage_manager.CopyMconSendToGPU(mcon_send, copy_stream);

  shared_data->SetMsgRecvPtr(control_thread_id, &mcon_recv);

  // Malloc auxiliary on device and copy it to device memory
  AuxiliaryManager::Allocate(
      conf->GetNumGPUControlThreads(),
      vcon.d_size, econ.d_size, mcon_recv.d_size, &auxiliary);
  gpu_storage_manager.CopyAuxiliaryToGPU(auxiliary, copy_stream);

  // Wait for reading and copying of each 'in' member of 'econ'
  WaitAndCopyEconToGPU();

  // Sync the copy_stream.
  checkCudaErrors(cudaStreamSynchronize(copy_stream));
  cudaStreamDestroy(copy_stream);

#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StartTimer(
      "Super step prepare " + Util::IToA(control_thread_id));
#endif
  PrepareSuperStep();
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StopTimer(
      "Super step prepare " + Util::IToA(control_thread_id));
#endif

#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StartTimer(
      "Super step run " + Util::IToA(control_thread_id));
#endif
  // Superstep
  if (conf->GetNumGPUControlThreads() == 1) {
    RunSuperStepsOnSingleGPU();
  } else {
    cout << "Only support single gpu at present!!" << endl;
    exit(0);
    // RunSuperStepsOnMultipleGPUs();
  }
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StopTimer(
      "Super step run " + Util::IToA(control_thread_id));
#endif
  writer->WriteOutput();

  VertexContentManager::Deallocate(&vcon);
  EdgeContentManager::Deallocate(&econ);
  MessageContentManager::Deallocate(&mcon_recv);
  MessageContentManager::Deallocate(&mcon_send);
  AuxiliaryManager::Deallocate(&auxiliary);
  // Never cudaDeviceReset in any thread because it will cause PROCESS behavior!

  // Signal main thread that all super steps are finished
  shared_data->SignalSuperStepFinished();
}

void GPUControlThreadData::WaitAndCopyVconToGPU() {
  while (true) {
    HostInVertexContent *buf = NULL;
    unsigned int size = 0;
    bool is_last = shared_data->WaitForVconBufferReady(
        control_thread_id, &buf, &size);
    // Copy to gpu.
    buf->CopyToDevice(vcon_copied_size, size, &vcon);
    vcon_copied_size += size;

    shared_data->SignalVconCopyFinish(control_thread_id);
    if (is_last) break;
  }
}

void GPUControlThreadData::WaitAndCopyEconToGPU() {
  while (true) {
    HostInEdgeContent *buf = NULL;
    unsigned int size = 0;
    bool is_last = shared_data->WaitForEconBufferReady(
        control_thread_id, &buf, &size);
    // Copy to gpu.
    buf->CopyToDevice(econ_copied_size, size, &econ);
    econ_copied_size += size;

    shared_data->SignalEconCopyFinish(control_thread_id);
    if (is_last) break;
  }
}

void GPUControlThreadData::PrepareSuperStep() {
#ifdef LAMBDA_DEBUG
  DBG_LOCK();
  cout << LAMBDA_HEADER << "GPUControlThreadData::PrepareSuperStep..." << endl;
  GlobalManager::DebugOutput(global);
  VertexContentManager::DebugOutput(vcon);
  EdgeContentManager::DebugOutput(econ);
  AuxiliaryManager::DebugOutput(auxiliary, vcon.d_size, econ.d_size, mcon_recv.d_size);
  cout.flush();
  DBG_UNLOCK();
#endif

  gpu_storage_manager.SingleGPUBuildIndexes(&vcon, &econ, &auxiliary);
  shared_data->GetGPUStatus(&local_gpu_status);
  gpu_storage_manager.CopyGPUStatusToGPU(local_gpu_status);

#ifdef LAMBDA_DEBUG
  DBG_LOCK();
  cout << LAMBDA_HEADER << "PrepareSuperStep finished." << endl;
  GlobalManager::DebugOutput(global);
  VertexContentManager::DebugOutput(vcon);
  EdgeContentManager::DebugOutput(econ);
  AuxiliaryManager::DebugOutput(auxiliary, vcon.d_size, econ.d_size, mcon_recv.d_size);
  cout.flush();
  DBG_UNLOCK();
#endif
}

void GPUControlThreadData::RunSuperStepsOnSingleGPU() {
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER
       << "GPUControlThreadData::RunSuperStepsOnSingleGPU... "
       << "max superstep: " << conf->GetMaxSuperstep() << endl;
  );
#endif
  const unsigned int max_superstep = conf->GetMaxSuperstep();
  unsigned int superstep = 0;
  for (; superstep < max_superstep; ++superstep) {
    if (!local_gpu_status.have_message && !local_gpu_status.alive) break;

#ifdef LAMBDA_PROFILING
    Singleton<Profiler>::GetInstance()->StartTimer(
        "Single-GPU copy message and clear status");
#endif

    if (local_gpu_status.have_message) {
#ifdef LAMBDA_ROLLING_MESSAGE_ARRAY
      std::swap(mcon_send, mcon_recv);
      gpu_storage_manager.CopyMconRecvToGPU(mcon_recv, 0);
      gpu_storage_manager.CopyMconSendToGPU(mcon_send, 0);
#else
      MessageContentManager::Copy(mcon_send, &mcon_recv);
#endif
    }
#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
    else {
      MessageContentManager::Clear(&mcon_recv);
    }
    MessageContentManager::Clear(&mcon_send);
#endif

    local_gpu_status.Clear();  // Set alive = 0 and have_message = 0.
    local_gpu_status.SetSuperStep(superstep);
    gpu_storage_manager.CopyGPUStatusToGPU(local_gpu_status);
#ifdef LAMBDA_PROFILING
    Singleton<Profiler>::GetInstance()->StopTimer(
        "Single-GPU copy message and clear status");
#endif

#ifdef LAMBDA_DEBUG
    DBG_LOCK();
    cout << LAMBDA_HEADER << "<<< Before UserCompute NO." << superstep << endl;
    VertexContentManager::DebugOutput(vcon);
    EdgeContentManager::DebugOutput(econ);
    AuxiliaryManager::DebugOutput(auxiliary, vcon.d_size, econ.d_size, mcon_recv.d_size);
    MessageContentManager::DebugOutput(mcon_send, true);
    MessageContentManager::DebugOutput(mcon_recv, false);
    cout.flush();
    DBG_UNLOCK();
#endif

#ifdef LAMBDA_PROFILING
    Singleton<Profiler>::GetInstance()->StartTimer("Single-GPU user compute");
#endif
    gpu_storage_manager.UserCompute(&vcon);  // user defined Compute
#ifdef LAMBDA_PROFILING
    Singleton<Profiler>::GetInstance()->StopTimer("Single-GPU user compute");
#endif

#ifdef LAMBDA_DEBUG
    DBG_LOCK();
    cout << LAMBDA_HEADER << "<<< After UserCompute NO." << superstep << endl;
    VertexContentManager::DebugOutput(vcon);
    EdgeContentManager::DebugOutput(econ);
    AuxiliaryManager::DebugOutput(auxiliary, vcon.d_size, econ.d_size, mcon_recv.d_size);
    MessageContentManager::DebugOutput(mcon_send, true);
    MessageContentManager::DebugOutput(mcon_recv, false);
    cout.flush();
    DBG_UNLOCK();
#endif

    // update local_gpu_status using device GPUStatus data
    gpu_storage_manager.GetGPUStatusFromGPU(&local_gpu_status);
  }
  cout << "Last super step number: " << superstep << endl;
}

void GPUControlThreadData::RunSuperStepsOnMultipleGPUs() {
// #ifdef LAMBDA_DEBUG
//   DBG_WRAP_COUT(
//   cout << LAMBDA_HEADER << "GPUControlThreadData::RunSuperStepsOnMultipleGPUs..." << endl;
//   cout << LAMBDA_HEADER << "max superstep: " << conf->GetMaxSuperstep() << endl;
//   );
// #endif
//   // Before starting the first round of super step, we need a synchronization
//   // for the end of the '0th' super step.
//   shared_data->WaitUntilAllGPUControlThreadsReachHere();
// 
//   const unsigned int max_superstep = conf->GetMaxSuperstep();
//   for (unsigned int superstep = 0; superstep < max_superstep; ++superstep) {
// #ifdef LAMBDA_DEBUG
//   DBG_WRAP_COUT(
//   cout << LAMBDA_HEADER << "Super step: " << superstep
//        << ", local_gpu_status.have_message: " << local_gpu_status.have_message
//        << ", local_gpu_status.alive:" << local_gpu_status.alive
//        << endl;
//   );
// #endif
//     if (!local_gpu_status.have_message && !local_gpu_status.alive) break;
// 
//     // Local shuffle and combine (if applicable) I
//     //
//     // TODO(laigd): If no combiner is available and all vertexes have sent =
//     // messages to all its neighbours (such as what PageRank does), we should
//     // do some optimization when doing the local shuffle, e.g. we can
//     // eliminate the comparison of d_is_full while sorting the messages.
//     const unsigned int *msg_per_gpu_begin = NULL, *msg_per_gpu_end = NULL;
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StartTimer(
//         "Multi-GPU local shuffle without combiner Ia " + Util::IToA(control_thread_id));
// #endif
//     gpu_storage_manager.MultiGPULocalShuffleWithoutCombinerIa(
//         local_gpu_status.have_message, &econ, &mcon_send, &auxiliary,
//         &msg_per_gpu_begin, &msg_per_gpu_end);
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StopTimer(
//         "Multi-GPU local shuffle without combiner Ia " + Util::IToA(control_thread_id));
// #endif
// 
//     // Global shuffle.
//     shared_data->ResetMsgRecvCount(control_thread_id);
//     shared_data->WaitUntilAllGPUControlThreadsReachHere();
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StartTimer(
//         "Multi-GPU global shuffle " + Util::IToA(control_thread_id));
// #endif
//     MultiGPUGlobalShuffle(msg_per_gpu_begin, msg_per_gpu_end);
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StopTimer(
//         "Multi-GPU global shuffle " + Util::IToA(control_thread_id));
// #endif
//     shared_data->WaitUntilAllGPUControlThreadsReachHere();
// 
//     const unsigned int num_msg_recv =
//         shared_data->GetMsgRecvCount(control_thread_id);
// #ifdef LAMBDA_DEBUG
//     DBG_LOCK();
//     cout << LAMBDA_HEADER
//          << "Global shuffle finished. Recv " << num_msg_recv << " messages."
//          << endl;
//     if (num_msg_recv > 0) mcon_recv.DebugOutputWithOffset(false, 0, num_msg_recv);
//     cout.flush();
//     DBG_UNLOCK();
// #endif
// 
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StartTimer(
//         "Multi-GPU local shuffle without combiner II " + Util::IToA(control_thread_id));
// #endif
//     // Local shuffle and combine (if applicable) II
//     gpu_storage_manager.MultiGPULocalShuffleWithoutCombinerII(
//         num_msg_recv, &vcon, &mcon_recv, &auxiliary);
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StopTimer(
//         "Multi-GPU local shuffle without combiner II " + Util::IToA(control_thread_id));
// #endif
// 
//     // Reset status.
//     local_gpu_status.Clear();  // Set alive = 0 and have_message = 0.
//     local_gpu_status.SetSuperStep(superstep);
//     gpu_storage_manager.CopyGPUStatusToGPU(local_gpu_status);
//     auxiliary.ClearForNextSuperStep(vcon.d_size, econ.d_size);
// 
// #ifdef LAMBDA_DEBUG
//     DBG_LOCK();
//     cout << LAMBDA_HEADER << "<<< Before UserCompute NO." << superstep << endl;
//     vcon.DebugOutput();
//     econ.DebugOutput();
//     auxiliary.DebugOutput(vcon.d_size, econ.d_size);
//     mcon_recv.DebugOutputWithOffset(false, 0, mcon_recv.d_size);
//     mcon_send.DebugOutputWithOffset(true, 0, mcon_send.d_size);
//     cout.flush();
//     DBG_UNLOCK();
// #endif
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StartTimer(
//         "User compute " + Util::IToA(control_thread_id));
// #endif
//     // User defined Compute.
//     gpu_storage_manager.UserCompute(&vcon);
// #ifdef LAMBDA_PROFILING
//     Singleton<Profiler>::GetInstance()->StopTimer(
//         "User compute " + Util::IToA(control_thread_id));
// #endif
// 
//     // Update local_gpu_status using all threads' device GPUStatus data.
//     gpu_storage_manager.GetGPUStatusFromGPU(&local_gpu_status);
//     shared_data->MergeGPUStatusAndWait(local_gpu_status);
//     shared_data->GetGPUStatus(&local_gpu_status);
// 
// #ifdef LAMBDA_DEBUG
//     DBG_LOCK();
//     cout << LAMBDA_HEADER << ">>> After UserCompute NO." << superstep << endl;
//     vcon.DebugOutput();
//     econ.DebugOutput();
//     auxiliary.DebugOutput(vcon.d_size, econ.d_size);
//     mcon_recv.DebugOutputWithOffset(false, 0, mcon_recv.d_size);
//     mcon_send.DebugOutputWithOffset(true, 0, mcon_send.d_size);
//     cout << LAMBDA_HEADER << "Super step NO." << superstep << " ends." << endl;
//     cout.flush();
//     DBG_UNLOCK();
// #endif
//   }
// 
//   // Wait for all super steps end.
//   shared_data->WaitUntilAllGPUControlThreadsReachHere();
}

void GPUControlThreadData::MultiGPUGlobalShuffle(
    const unsigned int *msg_per_gpu_begin,
    const unsigned int *msg_per_gpu_end) {
//   if (!local_gpu_status.have_message) return;
// 
//   for (int dest_gpu_id = 0;
//        dest_gpu_id < conf->GetNumGPUControlThreads();
//        ++dest_gpu_id) {
//     const unsigned int copy_size =
//         msg_per_gpu_end[dest_gpu_id] - msg_per_gpu_begin[dest_gpu_id];
// 
//     if (copy_size > 0) {
//       MessageContent *dest_mcon = NULL;
//       const unsigned int from_offset = msg_per_gpu_begin[dest_gpu_id];
//       const unsigned int dest_offset = shared_data->GetMsgRecvPtr(
//           dest_gpu_id, copy_size, &dest_mcon);
// #ifdef LAMBDA_DEBUG
//       DBG_LOCK();
//       cout << LAMBDA_HEADER
//            << "Copying message from gpu " << control_thread_id
//            << " to gpu " << dest_gpu_id
//            << ", copy size: " << copy_size
//            << ", src gpu offset: " << from_offset
//            << ", dest gpu offset: " << dest_offset
//            << endl;
//       mcon_send.DebugOutputWithOffset(true, from_offset, copy_size);
//       cout.flush();
//       DBG_UNLOCK();
// #endif
//       MessageContent::Copy(
//           mcon_send, from_offset, copy_size, dest_offset, dest_mcon);
//     }
//   }
//   cudaDeviceSynchronize();
}

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
