function p = bingham_pdf_3d(x,z1,z2,z3,v1,v2,v3,F)
% p = bingham_pdf_3d(x,z1,z2,z3,v1,v2,v3,F)

Z = F;
cos1 = dot(x,v1);
cos2 = dot(x,v2);
cos3 = dot(x,v3);
p = (1/Z)*exp(z1*cos1^2 + z2*cos2^2 + z3*cos3^2);
