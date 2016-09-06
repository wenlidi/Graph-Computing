// *****************************************************************************
// Filename:    user_graph_data_types.h
// Date:        2012-12-11 17:05
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

// Initial value for 'out' members will affect the 'InitOutMembers' function in
// both VertexContent and EdgeContenet, it can only be compile-time constants.
// If not specified, the default value is 0.
//
// Initial value for 'in' members will affect the 'Rand' functions in file
// generated_io_data_types.h if using random mode to run the program. The valid
// value includes:
// 1. RAND_VERTEX_ID
// 2. RAND_SMALL_UINT
// 4. RAND_MIDDLE_UINT
// 5. RAND_LARGE_UINT
// 6. RAND_FLOAT
// 7. RAND_FRACTION
// If not specified, the default value is 0 when using random mode to run.

struct Global {
  unsigned int source = RAND_VERTEX_ID;
};

struct Vertex {
  out unsigned int dist = ~0U;
  out unsigned int pre = ~0U;
};

struct Edge {
  in unsigned int weight = RAND_SMALL_UINT;
};

struct Message {
  unsigned int dist;
};
