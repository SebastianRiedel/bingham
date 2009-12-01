function x  = optimize_tetrahedron(v1, v2, v3, v4, step)
% Finds the minimal displacement x which makes (v1+x1,v2+x2,v3+x3,v4+x4) into
% an equilateral triangle.  Note: v1, v2, v3, and v4 are all assumed to be column
% vectors.

% Solve:   f'(x) + sum(z_i*g_i'(x)), g_i(x) = 0 for i=1:5
%   (where the z_i are lagrange multipliers)

if nargin < 5
   step = 1;
end

k = length(v1);

I = eye(k);
O = zeros(k);
i1 = 1:k;
i2 = k+1:2*k;
i3 = 2*k+1:3*k;
i4 = 3*k+1:4*k;

% jacobian of g_i(x) = Gi*x + ai
G1 = [O -2*I 2*I O; -2*I 2*I O O; 2*I O -2*I O ; O O O O];
a1 = [v3-v2; v2-v1; v1-v3; zeros(k,1)];

G2 = [O -2*I O 2*I; -2*I 2*I O O; O O O O; 2*I O O -2*I];
a2 = [v4-v2; v2-v1; zeros(k,1); v1-v4];

G3 = [2*I -2*I O O; -2*I O 2*I O; O 2*I -2*I O; O O O O];
a3 = [v1-v2; v3-v1; v2-v3; zeros(k,1)];

G4 = [O O O O; O O -2*I 2*I; O -2*I 2*I O; O 2*I O -2*I];
a4 = [zeros(k,1); v4-v3; v3-v2; v2-v4];

G5 = [O O O O; O 2*I -2*I O; O -2*I O 2*I; O O 2*I -2*I];
a5 = [zeros(k,1); v2-v3; v4-v2; v3-v4];

%a = [a1 a2 a3 a4 a5]

% initialize x = z1 = z2 = z3 = z4 = z5 = 0
x = zeros(4*k,1);
z1 = 0;
z2 = 0;
z3 = 0;
z4 = 0;
z5 = 0;

thresh = .00001;
iter = 100;
for i=1:iter
   
   %fprintf('.');
   
   % compute J, the full jacobian w.r.t. x, z1, z2
   I = eye(4*k);
   J = [2*I+z1*G1+z2*G2+z3*G3+z4*G4+z5*G5, G1*x+a1, G2*x+a2, G3*x+a3, G4*x+a4, G5*x+a5 ; ...
        (G1*x+a1)', 0, 0, 0, 0, 0 ; ...
        (G2*x+a2)', 0, 0, 0, 0, 0 ; ...
        (G3*x+a3)', 0, 0, 0, 0, 0 ; ...
        (G4*x+a4)', 0, 0, 0, 0, 0 ; ...
        (G5*x+a5)', 0, 0, 0, 0, 0 ];

   %det_J = det(J)
     
   % compute F, the system evaluated at the current x, z1, z2
   vx1 = v1 + x(i1);
   vx2 = v2 + x(i2);
   vx3 = v3 + x(i3);
   vx4 = v4 + x(i4);
   
   F = [2*x + z1*(G1*x+a1) + z2*(G2*x+a2) + z3*(G3*x+a3) + z4*(G4*x+a4) + z5*(G5*x+a5) ; ...
        norm(vx1-vx2)^2 - norm(vx1-vx3)^2 ; ...
        norm(vx1-vx2)^2 - norm(vx1-vx4)^2 ; ...
        norm(vx1-vx2)^2 - norm(vx2-vx3)^2 ; ...
        norm(vx2-vx3)^2 - norm(vx2-vx4)^2 ; ...
        norm(vx2-vx3)^2 - norm(vx3-vx4)^2 ];

   %input(':');
   
   % newton-raphson update
   y = J\-F;
   x = x + step*y(1:4*k);
   z1 = z1 + step*y(4*k+1);
   z2 = z2 + step*y(4*k+2);
   z3 = z3 + step*y(4*k+3);
   z4 = z4 + step*y(4*k+4);
   z5 = z5 + step*y(4*k+5);
   
   if norm(F) < thresh
      break;
   end
end

x = [x(i1) x(i2) x(i3) x(i4)];

fprintf('Optimized tetrahedron in %d iterations\n', i);


