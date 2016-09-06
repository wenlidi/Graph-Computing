#include <gflags/gflags.h>
#include "init.h"
#include "config.h"
#include "constants.h"

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Assume a maximum of 32 GPUs in a system configuration
  int available_device_id[kMaxNumGPUs];
  int num_available_device = GetGPUInfo(available_device_id);

  Config conf;
  SetConfigByCmdFlags(&conf);
  RunGPregel(conf, available_device_id, num_available_device);

  exit(true ? EXIT_SUCCESS : EXIT_FAILURE);
}
