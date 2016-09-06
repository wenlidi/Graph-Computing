function VsGPUHelp(i, gp, ba, ttl)
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
    '   R-MAT  ';
    '  Random  ';
    ' WikiTalk ';
    'RoadNet-CA';
    ];

subplot(1, 2, i);
bar(1:5:16, gp, 'BarWidth', 0.35, 'FaceColor', l1a);
hold on;
bar(3:5:18, ba, 'BarWidth', 0.35, 'FaceColor', l3c);

legend('GPregel', 'Basic', 'Location', 'NorthWest');
p = get(gca, 'Position') + [-0.01 0 0.02 0];
set(gca, 'Position', p);
set(gca, 'XLim', [-2 21]);
set(gca, 'XTick', 2:5:17);
set(gca, 'XTickLabel', label);
set(gca, 'YGrid', 'on');
title(ttl);
ylabel('Execution Time (ms)');

end