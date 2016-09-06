% origin result, may be not correct
%{
sssp_origin = [  413.657    396.799    75.041    294.904    ];
sssp_sorted = [  560.192    388.054    95.487    272.271    ];
bfs_origin = [    46.071     47.806    63.475    329.315    ];
bfs_sorted = [    66.329     46.639    81.670    229.045    ];
pg_origin = [   1353.85    1373.36   1071.71      83.065    ];
pg_sorted = [   1892.038   1388.314  1075.801     94.043    ];
bip_origin = [55.988];
bip_sorted = [44.086];
%}

%{
sssp_origin = [  413.702    250.477    25.426    329.470    ];
sssp_sorted = [   570.525    297.923    236.572    264.063    ];

% true result for bfs
% bfs_origin = [    46.061    2826.35    63.297    11895.8    ];
% bfs_sorted = [     68.348    42.540    82.873    33.834    ]; % must be an error!

pg_origin = [   1353.85    1373.36   1071.71      83.065    ];
pg_sorted = [   1892.038   1388.314  1075.801     94.043    ];
bip_origin = [55.988];
bip_sorted = [44.086];
%}

% new data @ 2013-3-28
%{
-------------- shortest_path --------------
rmat      --- origin:   973.826, sorted:  1188.383
rand      --- origin:   659.567, sorted:   649.737
wikitalk  --- origin:   374.578, sorted:   463.764
roadnetca --- origin:   516.905, sorted:   569.301

-------------- breadth_first_search --------------
rmat      --- origin:    52.458, sorted:    79.641
rand      --- origin:    49.161, sorted:    49.720
wikitalk  --- origin:    52.638, sorted:    64.109
roadnetca --- origin:   323.924, sorted:   292.538

-------------- page_rank --------------
rmat      --- origin:  2877.910, sorted:  3178.680
rand      --- origin:  2621.739, sorted:  2538.750
wikitalk  --- origin:  1264.257, sorted:  1282.738
roadnetca --- origin:    88.042, sorted:   176.746

-------------- bipartite_matching --------------
bip       --- origin:    76.168, sorted:    62.682
%}
sssp_origin = [973.826 659.567 374.578 516.905];
sssp_sorted = [1188.383 649.737 463.764 569.301];
bfs_origin = [52.458 49.161 52.638 323.924];
bfs_sorted = [79.641 49.720 64.109 292.538];
pg_origin = [2877.910 2621.739 1264.257 88.042];
pg_sorted = [3178.680 2538.750 1282.738 176.746];
bip_origin = [76.168];
bip_sorted = [62.682];

Test623Help(sssp_origin, sssp_sorted, 1, '(A) Single Source Shortest Path', 'Execution Time of Compute (ms)');
Test623Help(bfs_origin,  bfs_sorted,  2, '(B) Breadth First Search', 'Execution Time of Compute (ms)');
Test623Help(pg_origin,   pg_sorted,   3, '(C) Page Rank', 'Execution Time of Compute (ms)');

subplot(2, 2, 4);
bar(1, bip_origin, 'BarWidth', 0.9, 'FaceColor', 'g');
hold on;
bar(2, bip_sorted, 'BarWidth', 0.9, 'FaceColor', 'b');
legend('Basic Approach', 'Sorted');
set(gca, 'XLim', [-1 4]);
set(gca, 'XTick', 1.5);
set(gca, 'XTickLabel', ['BIP']);
set(gca, 'YGrid', 'on');
title('(D) Bipartite Matching');
% xlabel('Graphs');
ylabel('Execution Time of Compute (ms)');

