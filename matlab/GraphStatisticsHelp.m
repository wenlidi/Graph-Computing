function GraphStatisticsHelp(i, edges, vcnt, xlog, ylog, ttl)
subplot(4, 2, i);
bar(edges, vcnt);

p = get(gca, 'Position') + [-0.01 (5-floor((i+1)/2)*2)*0.01 0.02 0];
set(gca, 'Position', p);

set(gca, 'YGrid', 'on');

if xlog == 1
    set(gca, 'XScale', 'log');
end
if ylog == 1
    set(gca, 'YScale', 'log');
end

xl = get(gca, 'XLim');
xl(1) = 0;
set(gca, 'XLim', xl);

title(ttl);
xlabel('Number of Edges');
ylabel('Number of Vertexes');

end