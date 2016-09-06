function TestVsCPUHelp(m, i, ttl)
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

mf = fliplr(m);
cpu = mf(:, 3:4);
gpu = mf(:, 1:2);

subplot(2, 2, i);

bc = bar(1:4:13,       cpu, 'BarWidth', 0.35, 'BarLayout', 'stacked');
set(bc(1), 'FaceColor', l3c); % [0.15 0.6 0.9]);
set(bc(2), 'FaceColor', 'b'); % [0.05 0.05 0.97]);

hold on;

bg = bar(2.65:4:14.65, gpu, 'BarWidth', 0.35, 'BarLayout', 'stacked');
set(bg(1), 'FaceColor', l4a); % [1.0 0.8 0.8]);
set(bg(2), 'FaceColor', l2a); % 'r');

legend('CPU Run', 'CPU Prepare', 'GPregel Run', 'GPregel Prepare');
p = get(gca, 'Position') + [0 0 0.02 0.02] + (i <= 2)*[0 -0.02 0 0] + mod(i+1, 2)*[-0.02 0 0 0];
set(gca, 'Position', p);
set(gca, 'XLim', [-2 17]);
set(gca, 'XTick', 1.8:4:13.8);
set(gca, 'XTickLabel', label);
set(gca, 'YGrid', 'on');
title(ttl);
ylabel('Total Execution Time (ms)');
end

% set the color index of the first column
% ch = get(bc,'children');
% set(ch{1},'FaceVertexCData',[1;1;1;1;2;2;2;2;3;3;3;3;4;4;4;4;5;5;5;5;6;])