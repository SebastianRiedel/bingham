function S = bingham_scatter(B)
% S = bingham_scatter(B)

if B.Z==0  % uniform
    S = eye(B.d);
    return
end

d = B.d;
F = B.F;
dF = B.dF;
V = B.V;

v = bingham_mode(B);
sigma = 1 - sum(dF)/F;
S = sigma*v*v';

for i=1:d-1
    sigma = dF(i)/F;
    v = V(:,i);
    S = S + sigma*v*v';
end
