function q2 = qinv(q)
% q2 = qinv(q) -- invert a quaternion

if size(q,1) == 1
    q2 = [q(1), -q(2:4)];
else
    q2 = [q(1); -q(2:4)];
end
