function plot_axes(x,q,r)
%plot_axes(x,q,r) -- plots the axes defined by quaternion q at position x
%with axis length r

release_hold = ~ishold();

R = quaternion_to_rotation_matrix(q);
axis_colors = {'k-', 'g-', 'm-'};

plot3(x(1), x(2), x(3), 'g.', 'MarkerSize', 5);
hold on;
for i=1:3
    plot3([x(1) x(1)+r*R(1,i)], [x(2) x(2)+r*R(2,i)], [x(3) x(3)+r*R(3,i)], axis_colors{i}, 'LineWidth', 2);
end

if release_hold
    hold off;
end
