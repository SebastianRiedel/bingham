function plot_bingham_3d(B, Q)
% plot_bingham_3d(B, Q)

V = B.V;
Z = B.Z;
F = B.F;

clf;

subplot(2,1,1);
[SX,SY,SZ] = sphere(50);
n = size(SX,1);


% compute the marginal distribution of the axis 'u'
C = zeros(n);
for i=1:n
   for j=1:n
      u = [SX(i,j) SY(i,j) SZ(i,j)];
      for a=0:.1:2*pi
         q = [cos(a/2), sin(a/2)*u];
         C(i,j) = C(i,j) + bingham_pdf_3d(q, Z(1), Z(2), Z(3), V(:,1), V(:,2), V(:,3), F);
      end
   end
end

C = C./max(max(C));

surf(SX,SY,SZ,C, 'EdgeColor', 'none');
axis vis3d;
axis off;
colormap(.5*gray+.5);


if nargin >= 2
   n = size(Q,1);
   cmap = jet;
   P = zeros(1, n);
   for j=1:n
      P(j) = bingham_pdf(Q(j,:), B);
   end
   P = P./max(P);
   C = cmap(round(1+63*P), :);
   plot_quaternions(Q, C);
end
