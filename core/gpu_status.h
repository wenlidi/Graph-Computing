// *****************************************************************************
// Filename:    gpu_status.h
// Date:        2012-12-07 15:26
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef GPU_STATUS_H_
#define GPU_STATUS_H_

struct GPUStatus {
  bool alive;
  bool have_message;
  unsigned int superstep;

  void SetSuperStep(const unsigned int sups) {
    superstep = sups;
  }

  void CopyFrom(const GPUStatus &status) {
    alive = status.alive;
    have_message = status.have_message;
    superstep = status.superstep;
  }

  void MergeFrom(const GPUStatus &status) {
    if (status.alive) alive = true;
    if (status.have_message) have_message = true;
  }

  void Clear() {
    alive = false;
    have_message = false;
  }
};

#endif
