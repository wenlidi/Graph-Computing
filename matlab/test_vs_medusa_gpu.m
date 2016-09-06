% new data @ 2013-3-29, writing code manually according to paper "Accelerating large graph algorithms on the gpu using cuda"
%{
------------------------------------ sssp --------------------------------------
------ rmat ------
check result: correct, source: 247718, superstep: 52, gpu_duration: 377.566 ms.
check result: correct, source: 247718, superstep: 52, gpu_duration: 378.213 ms.
check result: correct, source: 247718, superstep: 52, gpu_duration: 378.261 ms.
check result: correct, source: 247718, superstep: 52, gpu_duration: 377.507 ms.
check result: correct, source: 247718, superstep: 52, gpu_duration: 379.05 ms.
check result: correct, source: 736048, superstep: 46, gpu_duration: 368.727 ms.
check result: correct, source: 736048, superstep: 46, gpu_duration: 368.309 ms.
check result: correct, source: 736048, superstep: 46, gpu_duration: 368.22 ms.
check result: correct, source: 736048, superstep: 46, gpu_duration: 368.921 ms.
check result: correct, source: 736048, superstep: 46, gpu_duration: 367.83 ms.
check result: correct, source: 936694, superstep: 41, gpu_duration: 335.466 ms.
check result: correct, source: 936694, superstep: 41, gpu_duration: 335.004 ms.
check result: correct, source: 936694, superstep: 41, gpu_duration: 335.329 ms.
check result: correct, source: 936694, superstep: 41, gpu_duration: 335.572 ms.
check result: correct, source: 936694, superstep: 41, gpu_duration: 336.357 ms.
------ rand ------
check result: correct, source: 99880, superstep: 34, gpu_duration: 307.876 ms.
check result: correct, source: 99880, superstep: 34, gpu_duration: 308.417 ms.
check result: correct, source: 99880, superstep: 34, gpu_duration: 308.038 ms.
check result: correct, source: 99880, superstep: 34, gpu_duration: 307.919 ms.
check result: correct, source: 99880, superstep: 34, gpu_duration: 308.162 ms.
check result: correct, source: 113275, superstep: 39, gpu_duration: 328.86 ms.
check result: correct, source: 113275, superstep: 39, gpu_duration: 329.06 ms.
check result: correct, source: 113275, superstep: 39, gpu_duration: 328.593 ms.
check result: correct, source: 113275, superstep: 39, gpu_duration: 328.909 ms.
check result: correct, source: 113275, superstep: 39, gpu_duration: 328.618 ms.
check result: correct, source: 946859, superstep: 34, gpu_duration: 308.298 ms.
check result: correct, source: 946859, superstep: 34, gpu_duration: 308.531 ms.
check result: correct, source: 946859, superstep: 34, gpu_duration: 308.297 ms.
check result: correct, source: 946859, superstep: 34, gpu_duration: 308.29 ms.
check result: correct, source: 946859, superstep: 34, gpu_duration: 309.698 ms.
------ wikitalk ------
check result: correct, source: 2379175, superstep: 28, gpu_duration: 177.02 ms.
check result: correct, source: 2379175, superstep: 28, gpu_duration: 177.327 ms.
check result: correct, source: 2379175, superstep: 28, gpu_duration: 175.795 ms.
check result: correct, source: 2379175, superstep: 28, gpu_duration: 176.23 ms.
check result: correct, source: 2379175, superstep: 28, gpu_duration: 174.724 ms.
check result: correct, source: 1729396, superstep: 34, gpu_duration: 215.772 ms.
check result: correct, source: 1729396, superstep: 34, gpu_duration: 221.048 ms.
check result: correct, source: 1729396, superstep: 34, gpu_duration: 218.09 ms.
check result: correct, source: 1729396, superstep: 34, gpu_duration: 221.602 ms.
check result: correct, source: 1729396, superstep: 34, gpu_duration: 219.664 ms.
check result: correct, source: 2357967, superstep: 30, gpu_duration: 148.632 ms.
check result: correct, source: 2357967, superstep: 30, gpu_duration: 144.933 ms.
check result: correct, source: 2357967, superstep: 30, gpu_duration: 148.693 ms.
check result: correct, source: 2357967, superstep: 30, gpu_duration: 146.222 ms.
check result: correct, source: 2357967, superstep: 30, gpu_duration: 147.825 ms.
------ roadnetca ------
check result: correct, source: 1787167, superstep: 708, gpu_duration: 508.704 ms.
check result: correct, source: 1787167, superstep: 708, gpu_duration: 509.479 ms.
check result: correct, source: 1787167, superstep: 708, gpu_duration: 503.816 ms.
check result: correct, source: 1787167, superstep: 708, gpu_duration: 508.042 ms.
check result: correct, source: 1787167, superstep: 708, gpu_duration: 504.514 ms.
check result: correct, source: 1743656, superstep: 677, gpu_duration: 518.894 ms.
check result: correct, source: 1743656, superstep: 677, gpu_duration: 518.811 ms.
check result: correct, source: 1743656, superstep: 677, gpu_duration: 518.927 ms.
check result: correct, source: 1743656, superstep: 677, gpu_duration: 518.253 ms.
check result: correct, source: 1743656, superstep: 677, gpu_duration: 518.492 ms.
check result: correct, source: 739769, superstep: 649, gpu_duration: 469.198 ms.
check result: correct, source: 739769, superstep: 649, gpu_duration: 469.237 ms.
check result: correct, source: 739769, superstep: 649, gpu_duration: 469.75 ms.
check result: correct, source: 739769, superstep: 649, gpu_duration: 465.884 ms.
check result: correct, source: 739769, superstep: 649, gpu_duration: 472.476 ms.

------------------------------------- bfs --------------------------------------
------ rmat ------
check result: correct, root: 247718, max_level: 7, superstep: 8, gpu_duration: 24.015 ms.
check result: correct, root: 247718, max_level: 7, superstep: 8, gpu_duration: 24.096 ms.
check result: correct, root: 247718, max_level: 7, superstep: 8, gpu_duration: 24.109 ms.
check result: correct, root: 247718, max_level: 7, superstep: 8, gpu_duration: 24.104 ms.
check result: correct, root: 247718, max_level: 7, superstep: 8, gpu_duration: 24.083 ms.
check result: correct, root: 736048, max_level: 7, superstep: 8, gpu_duration: 24.176 ms.
check result: correct, root: 736048, max_level: 7, superstep: 8, gpu_duration: 23.905 ms.
check result: correct, root: 736048, max_level: 7, superstep: 8, gpu_duration: 24.226 ms.
check result: correct, root: 736048, max_level: 7, superstep: 8, gpu_duration: 24.136 ms.
check result: correct, root: 736048, max_level: 7, superstep: 8, gpu_duration: 24.037 ms.
check result: correct, root: 936694, max_level: 7, superstep: 8, gpu_duration: 25.214 ms.
check result: correct, root: 936694, max_level: 7, superstep: 8, gpu_duration: 25.238 ms.
check result: correct, root: 936694, max_level: 7, superstep: 8, gpu_duration: 25.253 ms.
check result: correct, root: 936694, max_level: 7, superstep: 8, gpu_duration: 25.271 ms.
check result: correct, root: 936694, max_level: 7, superstep: 8, gpu_duration: 25.278 ms.
------ rand ------
check result: correct, root: 99880, max_level: 7, superstep: 8, gpu_duration: 36.065 ms.
check result: correct, root: 99880, max_level: 7, superstep: 8, gpu_duration: 36.143 ms.
check result: correct, root: 99880, max_level: 7, superstep: 8, gpu_duration: 36.084 ms.
check result: correct, root: 99880, max_level: 7, superstep: 8, gpu_duration: 36.291 ms.
check result: correct, root: 99880, max_level: 7, superstep: 8, gpu_duration: 36.011 ms.
check result: correct, root: 113275, max_level: 7, superstep: 8, gpu_duration: 37.206 ms.
check result: correct, root: 113275, max_level: 7, superstep: 8, gpu_duration: 36.914 ms.
check result: correct, root: 113275, max_level: 7, superstep: 8, gpu_duration: 37.134 ms.
check result: correct, root: 113275, max_level: 7, superstep: 8, gpu_duration: 37.138 ms.
check result: correct, root: 113275, max_level: 7, superstep: 8, gpu_duration: 37.02 ms.
check result: correct, root: 946859, max_level: 7, superstep: 8, gpu_duration: 37.531 ms.
check result: correct, root: 946859, max_level: 7, superstep: 8, gpu_duration: 37.639 ms.
check result: correct, root: 946859, max_level: 7, superstep: 8, gpu_duration: 37.633 ms.
check result: correct, root: 946859, max_level: 7, superstep: 8, gpu_duration: 37.473 ms.
check result: correct, root: 946859, max_level: 7, superstep: 8, gpu_duration: 37.494 ms.
------ wikitalk ------
check result: correct, root: 2379175, max_level: 7, superstep: 8, gpu_duration: 71.158 ms.
check result: correct, root: 2379175, max_level: 7, superstep: 8, gpu_duration: 70.731 ms.
check result: correct, root: 2379175, max_level: 7, superstep: 8, gpu_duration: 71.576 ms.
check result: correct, root: 2379175, max_level: 7, superstep: 8, gpu_duration: 71.19 ms.
check result: correct, root: 2379175, max_level: 7, superstep: 8, gpu_duration: 71.198 ms.
check result: correct, root: 1729396, max_level: 7, superstep: 8, gpu_duration: 76.507 ms.
check result: correct, root: 1729396, max_level: 7, superstep: 8, gpu_duration: 76.808 ms.
check result: correct, root: 1729396, max_level: 7, superstep: 8, gpu_duration: 76.461 ms.
check result: correct, root: 1729396, max_level: 7, superstep: 8, gpu_duration: 76.132 ms.
check result: correct, root: 1729396, max_level: 7, superstep: 8, gpu_duration: 75.979 ms.
check result: correct, root: 2357967, max_level: 7, superstep: 8, gpu_duration: 72.507 ms.
check result: correct, root: 2357967, max_level: 7, superstep: 8, gpu_duration: 74.885 ms.
check result: correct, root: 2357967, max_level: 7, superstep: 8, gpu_duration: 73.704 ms.
check result: correct, root: 2357967, max_level: 7, superstep: 8, gpu_duration: 73.403 ms.
check result: correct, root: 2357967, max_level: 7, superstep: 8, gpu_duration: 72.826 ms.
------ roadnetca ------
check result: correct, root: 1787167, max_level: 610, superstep: 611, gpu_duration: 184.299 ms.
check result: correct, root: 1787167, max_level: 610, superstep: 611, gpu_duration: 187.635 ms.
check result: correct, root: 1787167, max_level: 610, superstep: 611, gpu_duration: 181.567 ms.
check result: correct, root: 1787167, max_level: 610, superstep: 611, gpu_duration: 186.411 ms.
check result: correct, root: 1787167, max_level: 610, superstep: 611, gpu_duration: 182.472 ms.
check result: correct, root: 1743656, max_level: 579, superstep: 580, gpu_duration: 176.007 ms.
check result: correct, root: 1743656, max_level: 579, superstep: 580, gpu_duration: 173.094 ms.
check result: correct, root: 1743656, max_level: 579, superstep: 580, gpu_duration: 173.435 ms.
check result: correct, root: 1743656, max_level: 579, superstep: 580, gpu_duration: 173.648 ms.
check result: correct, root: 1743656, max_level: 579, superstep: 580, gpu_duration: 178.562 ms.
check result: correct, root: 739769, max_level: 562, superstep: 563, gpu_duration: 172.24 ms.
check result: correct, root: 739769, max_level: 562, superstep: 563, gpu_duration: 172.769 ms.
check result: correct, root: 739769, max_level: 562, superstep: 563, gpu_duration: 174.587 ms.
check result: correct, root: 739769, max_level: 562, superstep: 563, gpu_duration: 172.051 ms.
check result: correct, root: 739769, max_level: 562, superstep: 563, gpu_duration: 172.599 ms.

superstep time

sssp:
46.333 360.688
35.666 315.171
30.666 180.905
678.000 498.298

bfs:
8.000 24.476
8.000 36.918
8.000 73.671
584.666 177.425
%}

% rmat rand wikitalk roadnetca
% old data
%{
% gpregel = [171.872 58.161 72.191 340.802]; % original result
gpregel = [434.346 401.711 72.240 340.840];
medusa = (22.49 - [21.07 20.78 22.03 6.85]) / (22.49 - 18.88) * 500;
othergpu = (22.49 - [21.25 20.92 21.45 18.22]) / (22.49 - 18.88) * 500;
%}

sssp_gpregel = [983.629 667.217 377.553 583.167]; % new result @ 2013-3-28
sssp_base    = [360.688 315.171 180.905 498.298]; % new result @ 2013-3-29

medusa = (22.49 - [21.07 20.78 22.03 6.85]) / (22.49 - 18.88) * 500;
othergpu = (22.49 - [21.25 20.92 21.45 18.22]) / (22.49 - 18.88) * 500;
sssp_medusa  = medusa ./ othergpu .* sssp_base;

VsMedusaHelp(1, sssp_gpregel, sssp_medusa, sssp_base, '(A) Single Source Shortest Path');

bfs_gpregel = [54.481 51.178 53.464 381.020];
bfs_base    = [24.476 36.918 73.671 177.425];
VsGPUHelp(2, bfs_gpregel, bfs_base, '(B) Breadth First Search');
