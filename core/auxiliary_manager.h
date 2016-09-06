// *****************************************************************************
// Filename:    auxiliary_manager.h
// Date:        2013-01-07 11:40
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef AUXILIARY_MANAGER_H_
#define AUXILIARY_MANAGER_H_

struct AuxiliaryDeviceData;

class AuxiliaryManager {
 public:

  static void Allocate(
      const unsigned int num_gpus,
      const unsigned int vcon_array_size,
      const unsigned int econ_array_size,
      const unsigned int mcon_recv_array_size,
      AuxiliaryDeviceData *auxiliary);

  static void Deallocate(AuxiliaryDeviceData *auxiliary);

#ifdef LAMBDA_DEBUG
  static void DebugOutput(
      const AuxiliaryDeviceData &auxiliary,
      const unsigned int vcon_array_size,
      const unsigned int econ_array_size,
      const unsigned int mcon_recv_array_size);
#endif
};

#endif
