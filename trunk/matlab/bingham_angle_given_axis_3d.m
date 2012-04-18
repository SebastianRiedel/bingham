function theta = bingham_angle_given_axis_3d(B,v)
%theta = bingham_angle_given_axis_3d(B,v) -- Given an axis, v, compute the
%MLE angle with respect to B.

if B.d ~= 4
    fprintf('Error: bingham_angle_given_axis_3d() only supports 3D Binghams\n');
    return;
end

% make v a row vector
if size(v,2) == 1
    v = v';
end

z = B.Z;
% make z a column vector
if size(z,1) == 1
    z = z';
end

cos_theta = (B.V(1,:).^2 - (v * B.V(2:4,:)).^2) * z;
sin_theta = 2 * (B.V(1,:) .* (v * B.V(2:4,:))) * z;

theta = atan2(sin_theta, cos_theta);
