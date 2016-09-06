l1a = [204, 0, 0] ./ 255.0;  % red
l1b = [0, 0, 1];

l2a = [102, 102, 51] ./ 255.0;  % brown
l2b = [51, 102, 0] ./255.0;  % dark green
l2c = [0, 102, 204] ./255.0;  % blue
l2d = [255, 0, 255] ./255.0;  % pink

l3a = [51,204, 51] ./255.0;  % green
l3b = [0, 204, 0] ./255.0;  % green
l3c = [100, 255, 100] ./255.0;  % green

l4a = [255, 255, 153] ./255.0;  % yellow
l4b = [255, 255, 204] ./255.0;  % yellow
l4c = [204, 255, 204] ./255.0;  % eye protect

label = [
    '  SSSP  ';
    '  BFS   ';
    'PageRank';
    '   BM   ';
    ];

% rmat, rand, wiki, road
sssp = [ 1014.435  666.899  412.542  530.346];
bfs = [ 51.780 48.799 52.428 330.598];
pg = [ 2878.218 2622.245 1267.588 88.028];
bip = 76.211;

m = [sssp; bfs; pg;];

bm = bar(m);
set(bm(1), 'FaceColor', l1b);
set(bm(2), 'FaceColor', l3a);
set(bm(3), 'FaceColor', l4a);
set(bm(4), 'FaceColor', l2d);

hold on;
bar(4, bip, 'BarWidth', 0.22, 'FaceColor', l3c);

legend('R-MAT', 'Random', 'WikiTalk', 'RoadNet-CA', 'BIP', 'Location', 'NorthWest');
set(gca, 'XLim', [0.5 4.2]);
set(gca, 'XTick', 1:1:4);
set(gca, 'YTick', [0:100:500 1000:500:3000]);
set(gca, 'XTickLabel', label);
set(gca, 'YGrid', 'on');
title('Running Time of Different Algorithms on Different Graphs');
ylabel('Execution Time of Compute (ms)');