#!/bin/bash

./compiler/gpregel.py \
  -d shortest_path/user_graph_data_types.h \
  -t core/template/device_graph_data_types.h,core/template/generated_io_data_types.h,core/template/host_graph_data_types.h,core/template/host_in_graph_data_types.h,core/template/host_out_graph_data_types.h,core/template/user_api.h \
  -o instance/

