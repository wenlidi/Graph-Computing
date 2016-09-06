// *****************************************************************************
// Filename:    config.h
// Date:        2012-12-10 17:20
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef CONFIG_H_
#define CONFIG_H_

#include <string>

#include "hash_types.h"
#include "graph_types.h"
#include "writer_types.h"
#include "macro.h"

using std::string;

class Config {
 public:

  Config()
      : num_gpu_control_threads(0),
        num_reading_threads(0),
        max_superstep(0),
        num_vertex_for_random_graph(0),
        num_edge_for_random_graph(0),
        num_threads_per_block(0),
        single_gpu_id(0),
        dy_fenshuiling(0),
	      vertice_divide(1),
        blocksize_inner(128),
        graph_type(),
        input_file(),
        hash_type(HASH_MOD),
        output_file(),
        writer_type() {
  }

  // get/set of @num_gpu_control_threads
  unsigned int GetNumGPUControlThreads() const {
    return num_gpu_control_threads;
  }

  void SetNumGPUControlThreads(const unsigned int value) {
    num_gpu_control_threads = value;
  }

  // get/set of @num_reading_threads
  unsigned int GetNumReadingThreads() const {
    return num_reading_threads;
  }

  void SetNumReadingThreads(const unsigned int value) {
    num_reading_threads = value;
  }

  // get/set of @max_superstep
  unsigned int GetMaxSuperstep() const {
    return max_superstep;
  }

  void SetMaxSuperstep(const unsigned int value) {
    max_superstep = value;
  }

  // get/set of @num_vertex_for_random_graph
  unsigned int GetNumVertexForRandomGraph() const {
    return num_vertex_for_random_graph;
  }

  void SetNumVertexForRandomGraph(const unsigned int value) {
    num_vertex_for_random_graph = value;
  }

  // get/set of @num_edge_for_random_graph
  unsigned int GetNumEdgeForRandomGraph() const {
    return num_edge_for_random_graph;
  }

  void SetNumEdgeForRandomGraph(const unsigned int value) {
    num_edge_for_random_graph = value;
  }

  // get/set of @num_threads_per_block
  unsigned int GetNumThreadsPerBlock() const {
    return num_threads_per_block;
  }

  void SetNumThreadsPerBlock(const unsigned int value) {
    num_threads_per_block = value;
  }
  //get/set of @dy_fenshuiling
  unsigned int GetFenShuiLing() const {
    return dy_fenshuiling;
  }

  void SetFenShuiLing(const unsigned int value) {
cout << "dy_fenshuiling=" << value<<endl;
    dy_fenshuiling = value;
  }

    //get/set of @vertice_devide
  unsigned int GetBlocksizeInner() const {
    return blocksize_inner;
  }

    void SetBlocksizeInner(const unsigned int value) {
cout << "blocksize_inner=" << value<<endl;
    blocksize_inner = value;
  }

    //get/set of @vertice_devide
  unsigned int GetVerticeDivide() const {
    return vertice_divide;
  }


  void SetVerticeDivide(const unsigned int value) {
cout << "vertice_divide=" << value<<endl;
    vertice_divide = value;
  }

  // get/set of @single_gpu_id
  unsigned int GetSingleGPUId() const {
    return single_gpu_id;
  }

  void SetSingleGPUId(const unsigned int value) {
    single_gpu_id = value;
  }

  // get/set of @graph_type
  const GraphType& GetGraphType() const {
    return graph_type;
  }

  void SetGraphType(const GraphType &value) {
    graph_type = value;
  }

  // get/set of @input_file
  const string& GetInputFile() const {
    return input_file;
  }

  void SetInputFile(const string &value) {
    input_file = value;
  }

// get/set of sort_type
  const string& GetSortType() const {
    return sort_type;
  }

  void SetSortType(const string &value) {


    if(value == string("in_desc")||value ==string("in_asc")||value ==string("out_desc")||value==string("out_asc")||value==string("id") ){

     cout<< "sort type ok "<<endl;
    } else{
        cout << "illegal sort type \n sort_type is : in_desc, in_asc, out_desc, out_asc,id "<<endl;
        exit(1);
    }
    cout << "sort type is " << value<<endl;
    sort_type = value;
  }



  // get/set of @hash_type
  const HashType& GetHashType() const {
    return hash_type;
  }

  void SetHashType(const HashType &value) {
    hash_type = value;
  }

  const string& GetOutputFile() const {
    return output_file;
  }

  void SetOutputFile(const string &value) {
    output_file = value;
  }

  const WriterType& GetWriterType() const {
    return writer_type;
  }

  void SetWriterType(const WriterType &value) {
    writer_type = value;
  }
/*
  void DebugOutput() const {
    cout << "config {" << endl
         << "  num_gpu_control_threads: " << num_gpu_control_threads << endl
         << "  num_reading_threads: " << num_reading_threads << endl
         << "  max_superstep: " << max_superstep << endl
         << "  num_vertex_for_random_graph: " << num_vertex_for_random_graph << endl
         << "  num_edge_for_random_graph: " << num_edge_for_random_graph << endl
         << "  num_threads_per_block: " << num_threads_per_block << endl
         << "  single_gpu_id: " << single_gpu_id << endl
         << "  graph_type: " << graph_type.GetDescriptionString() << endl
         << "  input_file: " << input_file << endl
         << "  hash_type: " << (hash_type == HASH_MOD ? "mod" : "split") << endl
         << "  output_file: " << output_file << endl
         << "  writer_type: " << writer_type.GetDescriptionString() << endl
         << "}" << endl;
  }
*/
  void DebugOutput() const {
    cout << "config {" << endl
         << "  max_superstep: " << max_superstep << endl
         << "  input_file: " << input_file << endl
         << "  num_threads_per_block: " << num_threads_per_block << endl;
  }

 private:

  unsigned int num_gpu_control_threads;

  unsigned int num_reading_threads;

  unsigned int max_superstep;

  unsigned int num_vertex_for_random_graph;

  unsigned int num_edge_for_random_graph;

  unsigned int num_threads_per_block;  // Just used by user compute.
  

  unsigned int single_gpu_id;

  unsigned int dy_fenshuiling;
  unsigned int vertice_divide;
  unsigned int blocksize_inner;

  GraphType graph_type;

  string input_file;

  string sort_type;

  HashType hash_type;

  // Combiner setting

  // If writer_type is not *_FILE_WRITER, this member makes no sence.
  // Otherwise,
  // 1. if writer_type is SINGLE_FILE_WRITER, this value is exactly the file
  //    name of the single output file;
  // 2. if writer_type is MULTIPLE_FILE_WRITER, this value is the file name
  //    prefix of each shard.
  string output_file;

  WriterType writer_type;

  DISABLE_EVIL_DEFAULT_FUNCTIONS(Config);
};

#endif
