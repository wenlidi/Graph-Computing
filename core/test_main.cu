#include <vector>
#include <cstdlib>

#include <gflags/gflags.h>
#include "init.h"
#include "config.h"
#include "constants.h"

using std::vector;

struct VE {
  unsigned int v;
  unsigned int e;
};

int main(int argc, char **argv) {
  const unsigned int kNumTest = 100;

  google::ParseCommandLineFlags(&argc, &argv, true);
  int available_device_id[1] = { FLAGS_single_gpu_id };

  FLAGS_num_gpus = 1;
  FLAGS_max_superstep = 999999999;
  FLAGS_num_threads_per_block = 128;
  FLAGS_writer_type = "console_test";

  vector<VE> ve(kNumTest);
  srand(time(0));
  for (unsigned int i = 0; i < kNumTest; ++i) {
    ve[i].v = rand() % FLAGS_rand_num_vertex + 2;
    ve[i].e = rand() % FLAGS_rand_num_edge + 1;
  }

  Config conf;
  for (unsigned int i = 0; i < kNumTest; ++i) {
    FLAGS_rand_num_vertex = ve[i].v;
    FLAGS_rand_num_edge = ve[i].e;
    cout << endl;

    SetConfigByCmdFlags(&conf);
    RunGPregel(conf, available_device_id, 1);
  }

  exit(true ? EXIT_SUCCESS : EXIT_FAILURE);
}
