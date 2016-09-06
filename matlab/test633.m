small_origin=[ 27.546 26.714 26.106 26.454 27.483 26.078 27.459 27.867 29.953 31.624 32.723 33.686 34.420 35.486 36.269 37.065 ];
small_share=[ 21.173 21.516 21.187 22.717 23.280 25.089 25.527 26.145 27.879 29.318 30.310 31.501 32.171 33.039 33.973 34.903 ];
small_rolling=[ 23.374 22.486 22.853 22.585 22.659 22.614 22.735 22.584 22.246 23.684 23.596 22.589 22.961 23.256 22.859 22.882 ];

large_origin=[ 115.276 206.975 292.031 380.122 468.260 555.610 642.325 728.939 815.702 903.976 988.969 1077.138 1164.034 1249.808 1340.201 1421.515 ];
large_share=[ 114.532 201.502 286.882 375.063 463.093 549.861 635.738 722.265 810.331 898.412 985.136 1071.267 1156.714 1243.271 1330.643 1417.387 ];
large_rolling=[ 28.494 37.542 46.525 55.532 63.374 71.967 80.126 88.799 96.543 104.416 113.267 120.744 130.596 136.895 146.034 154.150 ];

subplot(1, 2, 1);
x = 10000:10000:160000;
plot(x, small_origin, 'ro-', x, small_share, 'gs-', x, small_rolling, 'bv-');
SetGreenLineColor(gca);

p = get(gca, 'Position') + [-0.03 0 0.04 0];
set(gca, 'Position', p);
set(gca, 'YGrid', 'on');
set(gca, 'XLim', [10000 160000]);
set(gca, 'XTick', 10000:10000:160000);
title('(A) Page Rank (10^4 Vertexes)');
xlabel('Number of Edges');
ylabel('Execution Time of Copying Message (ms)');
legend('Copy', 'Copy with Shared Array', 'Rolling Array', 'Location', 'NorthWest')

subplot(1, 2, 2);
x = 1000000:1000000:16000000;
plot(x, large_origin, 'ro-', x, large_share, 'gs-', x, large_rolling, 'bv-');
SetGreenLineColor(gca);

p = get(gca, 'Position') + [-0.01 0 0.04 0];
set(gca, 'Position', p);
set(gca, 'YGrid', 'on');
set(gca, 'XLim', [1000000 16000000]);
set(gca, 'XTick', 1000000:1000000:16000000);
title('(B) Page Rank (10^6 Vertexes)');
xlabel('Number of Edges');
ylabel('Execution Time of Copying Message (ms)');
legend('Copy', 'Copy with Shared Array', 'Rolling Array', 'Location', 'NorthWest')
