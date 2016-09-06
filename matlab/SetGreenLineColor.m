function SetGreenLineColor(handle)
lines = get(handle, 'Children');
for i = 1:size(lines)
    if get(lines(i), 'Color') == [0 1 0]
        set(lines(i), 'Color', [0 0.5 0]);
    end
end
end