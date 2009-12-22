function R = quaternionToRotationMatrix(q)
% R = quaternionToRotationMatrix(q)

a = q(1);
b = q(2);
c = q(3);
d = q(4);

R = [a*a + b*b - c*c - d*d,  2*b*c - 2*a*d,  2*b*d + 2*a*c  ;  ...
     2*b*c + 2*a*d,  a*a - b*b + c*c - d*d,  2*c*d - 2*a*b  ;  ...
     2*b*d - 2*a*c,  2*c*d + 2*a*b,  a*a - b*b - c*c + d*d  ];
