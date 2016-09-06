function Test623Help(origin, sorted, i, ttl, ylbl)
label = [
    '   R-MAT  ';
    '  Random  ';
    ' WikiTalk ';
    'RoadNet-CA';
    ];

subplot(2, 2, i);
bar(1:4:13, origin, 'BarWidth', 0.35, 'FaceColor', 'g');  % [0 0.5 0]);
hold on;
bar(2.65:4:14.65, sorted, 'BarWidth', 0.35, 'FaceColor', 'b');
legend('Basic Approach', 'Sorted');
set(gca, 'XLim', [-2 17]);
set(gca, 'XTick', 1.8:4:13.8);
set(gca, 'XTickLabel', label);
set(gca, 'YGrid', 'on');
title(ttl);
% xlabel(xlbl);
ylabel(ylbl);
end