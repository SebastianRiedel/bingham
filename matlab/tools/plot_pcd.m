function plot_pcd(pcd, style, options)
% plot_pcd(pcd)

if nargin < 2
    style = '.';
end

X = pcd.X;
Y = pcd.Y;
Z = pcd.Z;

plot3(X, Y, Z, style); %, 'MarkerSize', 20);

LINE_LENGTH_RATIO = .2; %.1;
LINE_SKIP_RATE = 5; %10;
mean_std = mean([std(X) std(Y) std(Z)]);
line_length = LINE_LENGTH_RATIO * mean_std;

plot_normals = 0;
plot_pcs = 0;
plot_vp = 0;
if nargin >= 3
   if strfind(options, 'n')
      plot_normals = 1;
   end
   if strfind(options, 'p')
      plot_pcs = 1;
   end
   if strfind(options, 'v')
       plot_vp = 1;
   end
end

% plot normals
if plot_normals
  hold on;
  NX = pcd.NX;
  NY = pcd.NY;
  NZ = pcd.NZ;
  r = line_length;
  for j=1:LINE_SKIP_RATE:length(NX)
     x = X(j); y = Y(j); z = Z(j);
     nx = NX(j); ny = NY(j); nz = NZ(j); 
     plot3([x x+r*nx], [y y+r*ny], [z z+r*nz], 'k-', 'LineWidth', 2);
  end
  hold off;
end

% plot principal curvatures
if plot_pcs
  hold on;
  PCX = pcd.PCX;
  PCY = pcd.PCY;
  PCZ = pcd.PCZ;
  r = line_length;
  for j=1:LINE_SKIP_RATE:length(PCX)
     x = X(j); y = Y(j); z = Z(j);
     pcx = PCX(j); pcy = PCY(j); pcz = PCZ(j); 
     plot3([x x+r*pcx], [y y+r*pcy], [z z+r*pcz], 'g-', 'LineWidth', 2);
  end
  for j=1:LINE_SKIP_RATE:length(PCX)
     x = X(j); y = Y(j); z = Z(j);
     pcx = PCX(j); pcy = PCY(j); pcz = PCZ(j); 
     nx = NX(j); ny = NY(j); nz = NZ(j); 
     pcx2 = ny*pcz - nz*pcy;
     pcy2 = nz*pcx - nx*pcz;
     pcz2 = nx*pcy - ny*pcx;
     plot3([x x+r*pcx2], [y y+r*pcy2], [z z+r*pcz2], 'm-', 'LineWidth', 2);
  end
  hold off;
end


% plot viewpoint
if plot_vp
    hold on;
    r = 10*line_length;
    vp = pcd.vp;
    R = quaternion_to_rotation_matrix(vp(4:7));
    [SX, SY, SZ] = sphere();
    surf(r/2*SX + vp(1), r/2*SY + vp(2), r/2*SZ + vp(3));
    plot3(vp(1)+[0,r]*R(1,1), vp(2)+[0,r]*R(2,1), vp(3)+[0,r]*R(3,1), 'k-', 'LineWidth', 3);
    plot3(vp(1)+[0,r]*R(1,2), vp(2)+[0,r]*R(2,2), vp(3)+[0,r]*R(3,2), 'g-', 'LineWidth', 3);
    plot3(vp(1)+[0,r]*R(1,3), vp(2)+[0,r]*R(2,3), vp(3)+[0,r]*R(3,3), 'm-', 'LineWidth', 3);
    hold off;
end

axis vis3d;
axis equal;

