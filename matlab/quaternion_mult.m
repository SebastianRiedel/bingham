function z = quaternion_mult(x,y)
% z = quaternion_mult(x,y)

a1 = x(1);
b1 = x(2);
c1 = x(3);
d1 = x(4);
a2 = y(1);
b2 = y(2);
c2 = y(3);
d2 = y(4);

z = x;  % just to match the dimensions

z(1) = a1*a2 - b1*b2 - c1*c2 - d1*d2;
z(2) = a1*b2 + b1*a2 + c1*d2 - d1*c2;
z(3) = a1*c2 - b1*d2 + c1*a2 + d1*b2;
z(4) = a1*d2 + b1*c2 - c1*b2 + d1*a2;
