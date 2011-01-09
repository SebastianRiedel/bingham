function [x s] = qksample(Q,X,q,r,x0,s0,w0)
% [x s] = qksample(Q,X,q,r) -- samples from a Quaternion Gaussian Kernel
% Local Likelihood model with data points Q,X, gaussian kernel with width r,
% and sample points q (in the rows).
% 
% [x s] = qksample(Q,X,q,r,x0,s0,w0)  -- samples with prior mean x0,
% covariance s0, and weight w0.


nq = size(Q,2);
nx = size(X,2);
N = size(Q,1);

if nargin < 7
    x0 = zeros(1,nx);
    s0 = zeros(nx);
    w0 = 0;
end

x = zeros(size(q,1),nx);     % sample means
s = zeros(nx,nx,size(q,1));  % sample covariance matrices
for i=1:size(q,1)
    qi = q(i,:);
    
    %dq = repmat(qi, [N 1]) - Q;
    %dq = sqrt(sum(dq.*dq, 2));
    dq = acos(abs(sum(repmat(qi, [N 1]).*Q, 2)));
    
    w = exp(-(dq/r).^2);        % weights
    wi = find(w>=max(w)/50);
    w = w(wi);                  % truncate small weights
    X = X(wi,:);
    wtot = sum(w) + w0;
    wx = repmat(w, [1 nx]).*X;
    x(i,:) = (x0*w0 + sum(wx)) / wtot;
    dx = X - repmat(x(i,:), [length(w) 1]);
    dwx = repmat(w, [1 nx]).*dx;
    S = 0;
    for j=1:size(dwx,1)
        S = S + dwx(j,:)'*dwx(j,:);
    end
    s(:,:,i) = (s0*w0 + S) / wtot;
end
