#include "init.h"

#include <ctime>
#include <cstdlib>
#include <string>
#include <fstream>
#include <ctime>
#include <cstdlib>

#include <gflags/gflags.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "config.h"
#include "reading_thread.h"
#include "reading_thread_data_types.h"
#include "gpu_control_thread.h"
#include "gpu_control_thread_data_types.h"
#include "multithreading.h"
#include "constants.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#endif

#ifdef LAMBDA_PROFILING
#include "singleton.h"
#include "profiler.h"
#endif

using std::string;
using std::cout;
using std::endl;
using std::ifstream;

DEFINE_int32(num_gpus, 1, "");
DEFINE_int32(single_gpu_id, 0, "");
DEFINE_int32(max_superstep, 999999999, "");
DEFINE_int32(num_threads_per_block, 64, "");
DEFINE_string(input_file, "", "");
DEFINE_string(graph_type, "simple", "");
DEFINE_string(hash_type, "mod", "");
DEFINE_string(output_file, "gpregel.out", "");
DEFINE_string(writer_type, "single", "");

// If FLAGS_graph_type is not "file" or "console" (i.e. generate the graph
// randomly), we use the following flags to control the graph.
DEFINE_int32(rand_num_vertex, 100, "");
DEFINE_int32(rand_num_edge, 1000, "");
DEFINE_int32(rand_num_reading_threads, 4, "");

namespace {

void StartThreadsAndWait(
    const Config &conf,
    const int *available_device_id,
    GPUControlThreadData *gpu_control_thread_data,
    ReadingThreadData *reading_thread_data) {
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->Clear();
#endif

  SharedData shared_data(&conf);

  cout << "Starting " << conf.GetNumGPUControlThreads()
       << " gpu control threads." << endl;

  if (conf.GetNumGPUControlThreads() == 1) {
    gpu_control_thread_data[0].Init(
        0, conf.GetSingleGPUId(), &conf, &shared_data);
    cutStartThread(GPUControlThread, &gpu_control_thread_data[0]);
  } else {
    for (int i = 0; i < conf.GetNumGPUControlThreads(); ++i) {
      gpu_control_thread_data[i].Init(
          i, available_device_id[i], &conf, &shared_data);
      cutStartThread(GPUControlThread, &gpu_control_thread_data[i]);
    }
  }

  cout << "Starting " << conf.GetNumReadingThreads()
       << " reading threads." << endl;
  if (conf.GetNumReadingThreads() == 1) {
    reading_thread_data[0].Init(0, &conf, &shared_data);
    reading_thread_data[0].Run();
  } else {
    for (int i = 0; i < conf.GetNumReadingThreads(); ++i) {
      reading_thread_data[i].Init(i, &conf, &shared_data);
      // TODO(laigd): Try to directly start reading_thread_data[i].Run()?
      cutStartThread(ReadingThread, &reading_thread_data[i]);
    }
  }

  shared_data.WaitForGlobalRead();
  shared_data.WaitForVconMemoryReady();
  shared_data.WaitForVconRead();
  shared_data.WaitForEconMemoryReady();
  shared_data.WaitForEconRead();
  shared_data.WaitForSuperStepFinished();

#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->Summary();
#endif
}

// If there only one input shard, then @out would be the filename, otherwise
// @out is the prefix before '@'.
unsigned int CheckInputFileName(const string &filename, string *out) {
  const size_t pos = filename.find('@');
  if (pos == string::npos) {
    (*out) = filename;
    return 1;
  } else {
    const string suffix = filename.substr(pos + 1);
    bool valid = true;
    if (suffix.length() == 0 || !(suffix[0] >= '1' && suffix[0] <= '9')) {
      valid = false;
    }
    if (valid) {
      for (size_t i = 1; i < suffix.length(); ++i) {
        if (suffix[i] <= '0' || suffix[i] >= '9') {
          valid = false;
          break;
        }
      }
    }
    if (!valid) {
      cout << "Invalid filename: " << filename << endl;
      exit(1);
    }
    unsigned int num_shard = std::atoi(filename.c_str());
    (*out) = filename.substr(0, pos);
    if (num_shard == 1) (*out) += "-01-of-01";
    return num_shard;
  }
}

}  // namespace

int GetGPUInfo(int *available_device_id) {
#ifdef LAMBDA_DEBUG
  for (int i = 0; i < 4; ++i) {
    available_device_id[i] = i;
  }
  return 4;
#else
  int num_gpus = 0, num_available_device = 0;
  checkCudaErrors(cudaGetDeviceCount(&num_gpus));
  cout << "Found " << num_gpus << " CUDA capable GPUs" << endl;

  if (num_gpus > kMaxNumGPUs) {
    cout << "simpleCallback only supports 32 GPU(s)" << endl;
  }

  for (int devid = 0; devid < num_gpus; ++devid) {
    int sm_version;
    cudaDeviceProp device_prop;
    cudaSetDevice(devid);
    cudaGetDeviceProperties(&device_prop, devid);
    sm_version = device_prop.major << 4 + device_prop.minor;
    cout << "GPU[" << devid << "] "
         << device_prop.name << " supports SM "
         << device_prop.major << "."
         << device_prop.minor << ", "
         << "unified addressing: "
         << (device_prop.unifiedAddressing ? "YES" : "NO!!!")
         << endl;
    if (sm_version >= 0x11 && device_prop.unifiedAddressing) {
      available_device_id[num_available_device++] = devid;
      if (num_available_device == kMaxNumGPUs) break;
    }
  }
  cudaDeviceReset();
  return num_available_device;
#endif
}

void SetConfigByCmdFlags(Config *conf) {
  conf->SetNumGPUControlThreads(FLAGS_num_gpus);
  conf->SetMaxSuperstep(FLAGS_max_superstep);
  conf->SetNumVertexForRandomGraph(FLAGS_rand_num_vertex);
  conf->SetNumEdgeForRandomGraph(FLAGS_rand_num_edge);
  conf->SetNumThreadsPerBlock(FLAGS_num_threads_per_block);
  conf->SetSingleGPUId(FLAGS_single_gpu_id);
  conf->SetGraphType(GraphType::GetGraphTypeFromString(FLAGS_graph_type));
  conf->SetWriterType(WriterType::GetWriterTypeFromString(FLAGS_writer_type));

  if (conf->GetGraphType() == GraphType::kGraphFromFile) {
    string input;
    const unsigned int num_shard = CheckInputFileName(FLAGS_input_file, &input);
    conf->SetInputFile(input);
    conf->SetNumReadingThreads(num_shard);
  } else if (conf->GetGraphType() == GraphType::kGraphFromConsole) {
    conf->SetNumReadingThreads(1);
  } else if (conf->GetGraphType() == GraphType::kSimpleGraph) {
    conf->SetNumReadingThreads(FLAGS_rand_num_reading_threads);
  } else if (conf->GetGraphType() == GraphType::kRMatGraph
             || conf->GetGraphType() == GraphType::kRandGraph) {
    conf->SetNumReadingThreads(1);
  }

  if (FLAGS_hash_type == "mod") {
    conf->SetHashType(HASH_MOD);
  // TODO(laigd): We need to define device pointers to enable hash functions.
  // } else if (FLAGS_hash_type == "split") {
  //   conf->SetHashType(HASH_SPLIT);
  } else {
    cout << "Invalid hash type!" << endl;
    exit(1);
  }

  if (conf->GetWriterType() == WriterType::kSingleFileWriter ||
      conf->GetWriterType() == WriterType::kMultipleFileWriter ||
      conf->GetWriterType() == WriterType::kFileTestWriter) {
    if (FLAGS_output_file.empty()) {
      cout << "Empty output file name!" << endl;
      exit(1);
    }
    conf->SetOutputFile(FLAGS_output_file);
  }

  conf->DebugOutput();
}

void RunGPregel(
    const Config &conf,
    const int *available_device_id,
    const int num_available_device) {
  // We use fixed seed instead of srand(time(0)) because we need the graph to be
  // determined each time we run the algorithm on different settings.
  srand(456416127);

  if (conf.GetNumGPUControlThreads() > num_available_device) {
    cout << "Not enough gpus for request!" << endl;
    exit(1);
  }

  if (conf.GetNumGPUControlThreads() == 1) {
    bool found = false;
    for (int i = 0; i < num_available_device; ++i) {
      if (available_device_id[i] == conf.GetSingleGPUId()) {
        found = true;
        break;
      }
    }
    if (!found) {
      cout << "Count not find available GPU of number "
           << conf.GetSingleGPUId() << endl;
      exit(1);
    }
  }

  cudaSetDevice(available_device_id[0]);

  GPUControlThreadData *gpu_control_thread_data =
      new GPUControlThreadData[conf.GetNumGPUControlThreads()];
  ReadingThreadData *reading_thread_data =
      new ReadingThreadData[conf.GetNumReadingThreads()];

  StartThreadsAndWait(
      conf,
      available_device_id,
      gpu_control_thread_data,
      reading_thread_data);

  delete[] gpu_control_thread_data;
  delete[] reading_thread_data;

  cudaDeviceReset();
}
