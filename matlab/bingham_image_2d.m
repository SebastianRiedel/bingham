function I = bingham_image_2d(z1,z2,v1,v2)
% I = bingham_image_2d(z1,z2,v1,v2)

w = 201;
h = 201;
r = 100;
cx = ceil(w/2);
cy = ceil(h/2);

F = bingham_F_2d(z1,z2,70);

I = zeros(h,w);

for yi=1:h
    %fprintf('.');
    y = (yi-cy)/r;
    for xi=1:w
        x = (xi-cx)/r;
        if x^2+y^2 > 1
            continue;
        end
        s = real([x y sqrt(1-x*x-y*y)]);
        I(yi,xi) = bingham_pdf_2d(s,z1,z2,v1,v2,F);
    end
end
%fprintf('\n');

I = I./max(max(I));
