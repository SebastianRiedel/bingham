function [Q Q2] = tofoo_sample(tofoo, pcd, n)
%Q = tofoo_sample(tofoo, pcd, n) -- monte carlo (metropolis hasting) sampling
%algorithm for object orientation given a tofoo model.  The proposal
%distribution is a mixture of orientation posteriors from single
%observations, and the target distribution is a function (geometric mean?)
%of the local orientation distributions over the full observed point cloud.
%
%The difficulty with computing the target distribution directly is that it
%is a nonlinear function of local orientation distributions, but we can
%compute the PDF in closed form for a given orientation.


npoints = size(pcd.X,1);
FCP = compute_feature_class_probs(tofoo, pcd, 1);
burn_in = 10;
sample_rate = 1; %10;

r = [];
num_accepts = 0;
Q = zeros(n,4);
Q2 = zeros(n,4);
for i=1:n*sample_rate+burn_in
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

    % sample a random acceptance threshold
    if num_accepts==0
        t2min = 0;
    else
        t2min = rand()*t*p2/p;
    end
    
    % compute target density for the given orientation 
    t2 = tofoo_cloud_pdf(r2, tofoo, pcd, FCP, t2min);

    % a = p*t2/(p2*t) >= rand()
    % t2 >= rand()*t*p2/p
    
    if num_accepts==0 || t2 > t2min
        num_accepts = num_accepts + 1;
        r = r2;
        p = p2;
        t = t2;
    end
    Q(i,:) = r;
    Q2(i,:) = r2;
    
    fprintf('.');
end
fprintf('\n');

accept_rate = num_accepts / (n*sample_rate + burn_in)

Q = Q(burn_in+1:sample_rate:end,:);
Q2 = Q2(burn_in+1:sample_rate:end,:);



