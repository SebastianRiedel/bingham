function p = bingham_pdf(x,B)
% p = bingham_pdf(x,B)

d = length(x);

if B.Z==0  % uniform
    p = 1/B.F;
    return
end

if d==3
   p = bingham_pdf_2d(x, B.Z(1), B.Z(2), B.V(:,1), B.V(:,2), B.F);
elseif d==4
   p = bingham_pdf_3d(x, B.Z(1), B.Z(2), B.Z(3), B.V(:,1), B.V(:,2), B.V(:,3), B.F);
else
   fprintf('Error: only 2-D and 3-D binghams supported in bingham_pdf()\n');
   p = [];
end
