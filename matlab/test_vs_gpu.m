sssp_gpregel = [983.629 667.217 377.553 583.167]; % new result @ 2013-3-28
sssp_base    = [360.688 315.171 180.905 498.298]; % new result @ 2013-3-29
VsGPUHelp(1, sssp_gpregel, sssp_base, '(A) Single Source Shortest Path');

bfs_gpregel = [54.481 51.178 53.464 381.020];
bfs_base    = [24.476 36.918 73.671 177.425];
VsGPUHelp(2, bfs_gpregel, bfs_base, '(B) Breadth First Search');