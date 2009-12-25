function q2 = qinv(q)
% q2 = qinv(q) -- invert a quaternion

q2 = [q(1) -q(2:4)];
