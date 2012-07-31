function I = pcd_to_image(pcd)
%I = pcd_to_image(pcd)

I = [];
if ~isfield(pcd, 'R') || ~isfield(pcd, 'G') || ~isfield(pcd, 'B')
    fprintf('Error: pcd has no color channels R,G,B\n');
    return
end
if ~isfield(pcd, 'width') || ~isfield(pcd, 'height') || pcd.width==1 || pcd.height==1
    fprintf('Error: pcd has no image dimensions\n');
    return
end

w = pcd.width;
h = pcd.height;

I = zeros(h,w,3);
I(:,:,1) = reshape(pcd.R, [w,h])';
I(:,:,2) = reshape(pcd.G, [w,h])';
I(:,:,3) = reshape(pcd.B, [w,h])';
I = I/256;
