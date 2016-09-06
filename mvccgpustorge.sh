CUDA_PATH="/usr/local/cuda-7.5"
CUDA_INC_PATH="${CUDA_PATH}/include"
CUDA_BIN_PATH="${CUDA_PATH}/bin"
CUDA_LIB_PATH="${CUDA_PATH}/lib64"
CUDA_COMMON_INC_PATH="${CUDA_PATH}/samples/common/inc"
GENCODE_FLAGS=${GENCODE_SM20}
CUDA_INCLUDES="-I${CUDA_INC_PATH} -I${CUDA_COMMON_INC_PATH}"
SPRNG_INCLUDES="-I./inc/sprng2.0-lite/include"
GENERAL_INCLUDES="${CUDA_INCLUDES} ${SPRNG_INCLUDES} -L${CUDA_LIB_PATH}	-lcudadevrt"
echo $GENERAL_INCLUDES

SPRNG_LIBRARIES="-L./inc/sprng2.0-lite/lib -lsprng -lm"
GENERAL_LIBRARIES="${SPRNG_LIBRARIES}"
GPREGEL_INCLUDES="-I${CORE_DIR} -I${INSTANCE_DIR} -I./out/shortest_path/default/instance/ -I./core"

NVCC_FLAGS="-D LAMBDA_DUMMY -m64 -O3 -rdc=true -gencode arch=compute_52,code=sm_52 ${GENERAL_INCLUDES} ${GENERAL_LIBRARIES} ${GPREGEL_INCLUDES}"
echo $NVCC_FLAGS

nvcc ${NVCC_FLAGS} -c ./core/modify/gpu_storage.cu -o ./out/shortest_path/default/bin/gpu_storageDYv3.o >> nvcclog1
