set -x

BUILD=${BUILD:-''}

if [ "${BUILD}" != '' ]; then
  BUILD_MARK=testmain USER_PATH=shortest_path make clean
  BUILD_MARK=testmain LAMBDA_DEBUG_FLAGS=LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT,LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS  \
    USER_PATH=shortest_path USER_FILE=./shortest_path/adjustable_heap.h make build
fi

l=21083
r=100000

for ((; l<r; )); do
  mid=$[(l+r)/2]

  ./out/shortest_path/testmain/main \
    --num_gpus=1 \
    --input_file= \
    --hash_type=mod \
    --rand_num_reading_threads=4 \
    \
    --output_file= \
    --writer_type=console_test \
    --max_superstep=999999999 \
    --num_threads_per_block=64 \
    --graph_type=rmat \
    --rand_num_vertex=${mid} \
    --rand_num_edge=$[${mid}*2] \
    > test-coalesced 2>&1

  res=`grep 'CUDA' test-coalesced`

  if [ "${res}" = "" ]; then
    l=$[mid+1]
  else
    r=$mid
    cp -f test-coalesced test-coalesced-backup
  fi
done
