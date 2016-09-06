% old data
%{
sssp_rmat=[    414.913    496.816    489.569    469.719    456.776    445.442    447.907    448.911    ];
sssp_rand=[    384.055    425.574    422.541    419.616    414.725    419.117    386.902    397.549    ];
sssp_wiki=[    74.940    77.556    79.302    77.954    78.913    79.257    88.983    88.107    ];
sssp_road=[    308.816    270.623    282.692    290.325    325.387    303.631    423.249   396.633    ];

bfs_rmat=[    19.860    52.633    52.026    50.538    49.895    50.834    56.724    55.551    ]; % + 270;
bfs_rand=[    47.305    50.623    49.547    49.573   49.666    50.134 47.607    48.411    ]; % + 3900;
bfs_wiki=[    63.380    65.141    66.784    66.227    66.258    67.335    75.674    74.600    ]; % + 65;
bfs_road=[    396.666    346.646    359.623    366.632    410.687 382.426    515.108    475.660    ]; % + 6400;

pg_rmat=[    1361.641    1732.050    1696.493    1613.117    1540.103    1440.548    1133.957    1202.377    ];
pg_rand=[    1362.654    1495.372    1488.795    1484.982    1441.611    1481.651    1283.206    1353.484    ];
pg_wiki=[    1069.773    1087.776    1082.219    1080.662    1077.860    1121.096    1295.366    1255.591    ];
pg_road=[    83.068    77.687    66.022    62.990    64.824    62.973    72.622    69.827    ];

bip=[    56.019    53.488    53.866    54.480    57.345    55.598    63.057    61.048    ];
%}

% new data @ 2013-3-28
sssp_rmat = [ 1014.435 1148.197 1139.770 1104.284 1077.445 1045.351 955.185 989.677];
sssp_rand = [ 666.899 702.490 697.818 691.995 680.252 687.378 648.638 659.689];
sssp_wiki = [ 412.542 433.510 436.513 428.796 416.932 404.925 453.702 443.677];
sssp_road = [ 530.346 496.498 506.298 518.260 561.772 534.351 702.557 663.886];

bfs_rmat = [ 51.780 61.266 62.016 59.968 59.223 57.619 57.861 58.677];
bfs_rand = [ 48.799 54.253 53.599 53.049 52.691 53.055 48.491 49.925];
bfs_wiki = [ 52.428 53.177 54.263 53.687 53.394 52.561 57.777 57.072];
bfs_road = [ 330.598 300.915 310.138 310.662 352.054 327.179 440.601 418.932];

pg_rmat = [ 2878.218 3140.790 3105.550 3036.645 2975.646 2904.810 2609.005 2719.961];
pg_rand = [ 2622.245 2661.118 2655.488 2654.833 2632.546 2652.179 2564.743 2598.641];
pg_wiki = [ 1267.588 1293.128 1257.173 1277.726 1246.022 1204.275 1347.208 1284.335];
pg_road = [ 88.028 89.529 85.610 85.181 77.044 86.270 75.514 74.121];

bip = [76.211 79.533 79.580 78.980 80.722 79.868 82.820 82.070 ];

% ------------------------ start ----------------------------

x = 128:128:1024;

subplot(2, 2, 1);
plot(x, sssp_rmat, 'ro:', x, sssp_rand, 'gs:', x, sssp_wiki, 'bx-', x, sssp_road, 'k*-');
SetGreenLineColor(gca);
set(gca, 'XLim', [128 1024]);
set(gca, 'XTick', 128:128:1024);
% set(gca, 'YGrid', 'on');
title('(A) Single Source Shortest Path');
xlabel('Block Size');
ylabel('Execution Time of Compute (ms)');
legend('R-MAT', 'Random', 'WikiTalk', 'RoadNet-CA', 'Location', 'East')

subplot(2, 2, 2);
plot(x, bfs_rmat, 'ro:', x, bfs_rand, 'gs:', x, bfs_wiki, 'bx-', x, bfs_road, 'k*-');
SetGreenLineColor(gca);
set(gca, 'XLim', [128 1024]);
set(gca, 'XTick', 128:128:1024);
% set(gca, 'YGrid', 'on');
title('(B) Breadth First Search');
xlabel('Block Size');
ylabel('Execution Time of Compute (ms)');
legend('R-MAT', 'Random', 'WikiTalk', 'RoadNet-CA', 'Location', 'East')

subplot(2, 2, 3);
plot(x, pg_rmat, 'ro:', x, pg_rand, 'gs:', x, pg_wiki, 'bx-', x, pg_road, 'k*-');
SetGreenLineColor(gca);
set(gca, 'XLim', [128 1024]);
set(gca, 'XTick', 128:128:1024);
% set(gca, 'YGrid', 'on');
title('(C) Page Rank');
xlabel('Block Size');
ylabel('Execution Time of Compute (ms)');
legend('R-MAT', 'Random', 'WikiTalk', 'RoadNet-CA', 'Location', 'East')

subplot(2, 2, 4);
plot(x, bip, 'mo-');
set(gca, 'XLim', [128 1024]);
set(gca, 'XTick', 128:128:1024);
% set(gca, 'YGrid', 'on');
title('(D) Bipartite Matching');
xlabel('Block Size');
ylabel('Execution Time of Compute (ms)');
legend('BIP', 'Location', 'East')