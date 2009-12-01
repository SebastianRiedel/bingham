function p = bingham_pdf_2d(x,z1,z2,v1,v2,F)
% p = bingham_pdf_2d(x,z1,z2,v1,v2,F)

iter = 20;

Z = F; %bingham_1F1_2d(z1,z2,iter);
cos1 = dot(x,v1);
cos2 = dot(x,v2);
p = (1/Z)*exp(z1*cos1^2 + z2*cos2^2);
