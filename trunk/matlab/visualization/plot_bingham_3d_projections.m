function plot_bingham_3d_projections(V, Z, F)
% plot_bingham_3d_projections(V, Z, F)


[SX,SY,SZ] = sphere(30);
n = size(SX,1);

C1 = zeros(n);
C2 = zeros(n);
C3 = zeros(n);
C4 = zeros(n);

for i=1:n
   for j=1:n
      q1 = [0 SX(i,j) SY(i,j) SZ(i,j)];
      q2 = [SX(i,j) 0 SY(i,j) SZ(i,j)];
      q3 = [SX(i,j) SY(i,j) 0 SZ(i,j)];
      q4 = [SX(i,j) SY(i,j) SZ(i,j) 0];
      C1(i,j) = bingham_pdf_3d(q1, Z(1), Z(2), Z(3), V(:,1), V(:,2), V(:,3), F);
      C2(i,j) = bingham_pdf_3d(q2, Z(1), Z(2), Z(3), V(:,1), V(:,2), V(:,3), F);
      C3(i,j) = bingham_pdf_3d(q3, Z(1), Z(2), Z(3), V(:,1), V(:,2), V(:,3), F);
      C4(i,j) = bingham_pdf_3d(q4, Z(1), Z(2), Z(3), V(:,1), V(:,2), V(:,3), F);
   end
end

C1 = C1./max(max(C1));
C2 = C2./max(max(C2));
C3 = C3./max(max(C3));
C4 = C4./max(max(C4));

subplot(2,2,1);
surf(SX,SY,SZ,C1, 'EdgeColor', 'none');
axis vis3d;
subplot(2,2,2);
surf(SX,SY,SZ,C2, 'EdgeColor', 'none');
axis vis3d;
subplot(2,2,3);
surf(SX,SY,SZ,C3, 'EdgeColor', 'none');
axis vis3d;
subplot(2,2,4);
surf(SX,SY,SZ,C4, 'EdgeColor', 'none');
axis vis3d;
%colormap(.5*gray+.5);
