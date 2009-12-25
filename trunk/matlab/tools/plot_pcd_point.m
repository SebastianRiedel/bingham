function plot_pcd_point(pcd, i, options, style)
% plot_pcd_point(pcd, i, options, style)

plot_normals = 0;
plot_pcs = 0;
if nargin >= 3
   if strfind(options, 'n')
      plot_normals = 1;
   end
   if strfind(options, 'p')
      plot_pcs = 1;
      plot_normals = 1;
   end
end

if nargin < 4
    style = 'r.';
end

LINE_LENGTH_RATIO = 1;
mean_std = mean([std(pcd.X) std(pcd.Y) std(pcd.Z)]);
r = LINE_LENGTH_RATIO * mean_std;

x = pcd.X(i);
y = pcd.Y(i);
z = pcd.Z(i);

plot3(x, y, z, style, 'MarkerSize', 5, 'LineWidth', 5);
hold on;
   
if plot_normals
    nx = pcd.NX(i);
    ny = pcd.NY(i);
    nz = pcd.NZ(i);
    plot3([x x+r*nx], [y y+r*ny], [z z+r*nz], 'k-', 'LineWidth', 2);
end
   
% plot principal curvatures
if plot_pcs
    pcx = pcd.PCX(i);
    pcy = pcd.PCY(i);
    pcz = pcd.PCZ(i);
    plot3([x x+r*pcx], [y y+r*pcy], [z z+r*pcz], 'g-', 'LineWidth', 2);
    pcx2 = ny*pcz - nz*pcy;
    pcy2 = nz*pcx - nx*pcz;
    pcz2 = nx*pcy - ny*pcx;
    plot3([x x+r*pcx2], [y y+r*pcy2], [z z+r*pcz2], 'm-', 'LineWidth', 2);
end

hold off;
