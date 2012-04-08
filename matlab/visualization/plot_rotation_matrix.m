function plot_rotation_matrix(R)
% plot_rotation_matrix(R)

plot3([0 R(1,1)], [0 R(2,1)], [0 R(3,1)], 'k-', 'LineWidth', 2);
hold on;
plot3([0 R(1,2)], [0 R(2,2)], [0 R(3,2)], 'g-', 'LineWidth', 2);
plot3([0 R(1,3)], [0 R(2,3)], [0 R(3,3)], 'm-', 'LineWidth', 2);
hold off;
axis([-1 1 -1 1 -1 1]);
