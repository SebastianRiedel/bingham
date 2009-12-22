function plot_quaternions(Q, C)
% plot_quaternions(Q) -- plots quaternions as an axis/angle chart (Q is
% n-by-4)

%clf;

subplot(2,1,1);
%[SX,SY,SZ] = sphere(30);
%surf(SX,SY,SZ, 'EdgeColor', 'none');
axis vis3d;
%colormap(.5*gray+.5);
hold on;

subplot(2,1,2);
axis([0 2*pi 0 1]);
hold on;

for i=1:size(Q,1)

   a = 2*acos(Q(i,1));
   if abs(sin(a/2)) < 1e-10   % no rotation, so pick an arbitrary axis
      fprintf('.');
      v = [1 0 0];
   else
      v = Q(i,2:4) / sin(a/2);
   end
   
   cmap = jet;
   
   if nargin < 2
      c = cmap(round(1+63*a/(2*pi)), :);
   else
      c = C(i,:);
   end
   
   % plot axis
   subplot(2,1,1);
   plot3(v(1), v(2), v(3), '.', 'Color', c);
   
   % plot angle
   subplot(2,1,2);
   plot(a, 0, 'o', 'Color', c);
end

subplot(2,1,1);
hold off;
subplot(2,1,2);
hold off;


fprintf('\n');
