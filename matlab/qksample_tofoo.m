function [x_mean x_cov] = qksample_tofoo(q, c, tofoo)
% [x_mean x_cov] = qksample_tofoo(q, c, tofoo) -- returns p(x|q) from a Quaternion Gaussian
% Kernel Local Likelihood model, with model (tofoo), and feature class (c).


% get non-parametric model, (X,Q)
I = find(tofoo.pcd.L == c);
Q = tofoo.pcd.Q(I,:,:);
X = [tofoo.pcd.X(I), tofoo.pcd.Y(I), tofoo.pcd.Z(I)];
Q = [Q(:,:,1); Q(:,:,2)];
X = [X; X];

% get prior
x0 = zeros(1,3);
X0 = [tofoo.pcd.X, tofoo.pcd.Y, tofoo.pcd.Z];
s0 = sqrt(mean(sum(X0.^2,2))) * eye(3);
w0 = 2;

% compute p(x|q)
r = .1;
[x_mean x_cov] = qksample(Q,X,q,r,x0,s0,w0);
