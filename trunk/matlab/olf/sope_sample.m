function [XQ W] = sope_sample(tofoo, pcd, n)
%XQ = sope_sample(tofoo, pcd, n) -- importance sampling
%algorithm for object pose given a tofoo model.

hard_assignment = 1;

num_samples = 5;
lambda = .5;

npoints = size(pcd.X,1);
FCP = compute_feature_class_probs(tofoo, pcd, hard_assignment);

XQ = zeros(n,7);
W = zeros(1,n);
for i=1:n
    % sample a point feature
    j = ceil(rand()*npoints);
    f = pcd.F(j,:);
    if rand() < .5
        q = pcd.Q(j,1:4);
    else
        q = pcd.Q(j,5:8);
    end
    
    % compute the model orientation posterior given the feature
    BMM = tofoo_posterior(tofoo, q, f);
    
    % sample an orientation from the proposal distribution
    r2 = bingham_mixture_sample(BMM.B, BMM.W, 1);
    p2 = bingham_mixture_pdf(r2, BMM.B, BMM.W);
    %r2_err = acos(abs(r2(1)^2 - r2(2)^2 - r2(3)^2 + r2(4)^2))'

    % sample from the proposal distribution of position given orientation
    xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];
    c = find(FCP(j,:));
    q2 = quaternion_mult(q, qinv(r2));
    [x_mean x_cov] = qksample_tofoo(q2,c,tofoo);
    x0 = mvnrnd(x_mean, x_cov);
    R = quaternionToRotationMatrix(r2);
    x2 = (xj' - R*x0')';
    p2 = p2*mvnpdf(x0, x_mean, x_cov);
    
    % compute target density for the given orientation 
    t2 = sope_cloud_pdf(x2, r2, tofoo, pcd, FCP, num_samples, lambda);
        
    XQ(i,:) = [x2 r2];
    W(i) = t2; %/p2;
    
    fprintf('.');
end
fprintf('\n');


[W I] = sort(W,'descend');
XQ = XQ(I,:);
W = W/sum(W);




