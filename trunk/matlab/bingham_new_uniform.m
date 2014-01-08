function B = bingham_new_uniform(d)
%B = bingham_new_uniform(d)

B = struct('d',d);
B.Z = zeros(1,d-1);
B.V = eye(d);
B.V = B.V(:,2:end);
[B.F B.dF] = bingham_F(B.Z);
