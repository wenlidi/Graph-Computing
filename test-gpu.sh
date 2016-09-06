source ./papertest/vid_with_most_out_edge.sh
source ./papertest/make_helper_code.sh
source ./papertest/helper_script.sh
set +x

TESTCASE_DIR=papertest/test_cases

function Run_SSSP_BFS() {
  echo ------ ${GRAPH_SUFFIX} ------
  SHORT_INPUT_FILE_NAME=${GRAPH_SUFFIX}-raw.graph
  ORIGIN=${TESTCASE_DIR}/${SHORT_INPUT_FILE_NAME}
  
echo  ${GRAPH_GEN} ${ORIGIN} ${ALGORITHM} 1  
  for ((i=0; i<1; i=i+1)); do
    for ((j=0; j<1; j=j+1)); do
       ${GRAPH_GEN} ${ORIGIN} ${ALGORITHM} 1|  ${HELPER_OUT_DIR}/${EXEFILE} 1 
    done
  done
}

function RunPageRank() {
  for GRAPH_SUFFIX in rmat rand wikitalk roadnetca; do
    ORIGIN=${TESTCASE_DIR}/${GRAPH_SUFFIX}-raw.graph
    echo ------ ${GRAPH_SUFFIX} ------

    for ((i=0; i<3; i=i+1)); do
      ${GRAPH_GEN} ${ORIGIN} ${ALGORITHM} | ${HELPER_OUT_DIR}/pg_gpu.out
    done
  done
}

SRC_DIR=papertest/gpu_test_code
INC_DIR=papertest/cpu_test_code
#NVCC="/usr/local/cuda-5.0/bin/nvcc -m64 -O3 -gencode arch=compute_20,code=sm_20 -I/usr/local/cuda-5.0/include -I/usr/local/cuda-5.0/samples/common/inc -I./${INC_DIR}"
NVCC="/usr/local/cuda-7.5/bin/nvcc -m64 -O3 -gencode arch=compute_52,code=sm_52 -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.5/samples/common/inc -I./${INC_DIR}"
echo ------------------------------------ sssp --------------------------------------
${NVCC} -c ${INC_DIR}/sssp.cc -o ${HELPER_OUT_DIR}/sssp.o
${NVCC} -c ${SRC_DIR}/sssp_gpu.cu -o ${HELPER_OUT_DIR}/sssp_gpu.o
${NVCC} -o ${HELPER_OUT_DIR}/sssp_gpu.out ${HELPER_OUT_DIR}/sssp.o ${HELPER_OUT_DIR}/sssp_gpu.o
ALGORITHM=sssp
EXEFILE=sssp_gpu.out
Run_SSSP_BFS_OnAllGraph

echo ------------------------------------- bfs --------------------------------------
${NVCC} -c ${INC_DIR}/bfs.cc -o ${HELPER_OUT_DIR}/bfs.o
${NVCC} -c ${SRC_DIR}/bfs_gpu.cu -o ${HELPER_OUT_DIR}/bfs_gpu.o
${NVCC} -o ${HELPER_OUT_DIR}/bfs_gpu.out ${HELPER_OUT_DIR}/bfs.o ${HELPER_OUT_DIR}/bfs_gpu.o
ALGORITHM=bfs
EXEFILE=bfs_gpu.out
Run_SSSP_BFS_OnAllGraph

echo ------------------------------------- pg --------------------------------------

# ${NVCC} ${SRC_DIR}/page_rank.cc -o ${HELPER_OUT_DIR}/pg.out

# ALGORITHM=pg
# RunPageRank
