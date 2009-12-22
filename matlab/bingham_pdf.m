function p = bingham_pdf(x,V,Z,F)
% p = bingham_pdf(x,V,Z,F)

d = length(x);

if d==3
   p = bingham_pdf_2d(x, Z(1), Z(2), V(:,1), V(:,2), F);
elseif d==4
   p = bingham_pdf_3d(x, Z(1), Z(2), Z(3), V(:,1), V(:,2), V(:,3), F);
else
   fprintf('Error: only 2-D and 3-D binghams supported in bingham_pdf()\n');
   p = [];
end
