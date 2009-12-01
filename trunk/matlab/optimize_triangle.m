function x  = optimize_triangle(v1, v2, v3)
% Finds the minimal displacement x which makes (v1+x1,v2+x2,v3+x3) into
% an equilateral triangle.  Note: v1, v2, and v3 are all assumed to be column
% vectors.

% Solve:   f'(x) + z1*g'(x) + z2*h'(x) = 0, g(x) = 0, h(x) = 0
%   (where z1 and z2 are lagrange multipliers)

k = length(v1);

I = eye(k);
O = zeros(k);
i1 = 1:k;
i2 = k+1:2*k;
i3 = 2*k+1:3*k;

% jacobian of g(x) = Gx + a
G = [O -2*I 2*I ; -2*I 2*I O ; 2*I O -2*I];
a = [v3-v2 ; v2-v1; v1-v3];

% jacobian of h(x) = Hx + b
H = [2*I -2*I O ; -2*I O 2*I ; O 2*I -2*I];
b = [v1-v2 ; v3-v1; v2-v3];

% initialize x = z1 = z2 = 0
x = zeros(3*k,1);
z1 = 0;
z2 = 0;

thresh = .00001;
iter = 100;
for i=1:iter
   
   % compute J, the full jacobian w.r.t. x, z1, z2
   I = eye(3*k);
   J = [2*I+z1*G+z2*H, G*x+a, H*x+b ; (G*x+a)', 0, 0 ; (H*x+b)', 0, 0];
   
   % compute F, the system evaluated at the current x, z1, z2
   vx1 = v1 + x(i1);
   vx2 = v2 + x(i2);
   vx3 = v3 + x(i3);
   
   %v = [v1 v2 v3];
   %vx = [vx1 vx2 vx3];
   %plot(v(1,[1:end 1]), v(2,[1:end 1]), 'r-');
   %hold on;
   %plot(vx(1,[1:end 1]), vx(2,[1:end 1]));
   %hold off;
   %axis([0 1 0 1]);
   
   F = [2*x + z1*(G*x+a) + z2*(H*x+b) ; norm(vx1-vx2)^2 - norm(vx1-vx3)^2 ; norm(vx1-vx2)^2 - norm(vx2-vx3)^2];

   %input(':');
   
   % newton-raphson update
   y = J\-F;
   x = x + y(1:3*k);
   z1 = z1 + y(3*k+1);
   z2 = z2 + y(3*k+2);
   
   if norm(F) < thresh
      break;
   end
end

x = [x(i1) x(i2) x(i3)];

fprintf('Optimized triangle in %d iterations\n', i);


