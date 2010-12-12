function p = tofoo_cloud_pdf(q, tofoo, pcd, FCP, pmin, num_samples)
%p = tofoo_cloud_pdf(q, tofoo, pcd, FCP, pmin, num_samples) -- computes the
%pdf of a point cloud given a tofoo model and a model pose, q, with prob.
%threshold pmin (for pruning).

if nargin < 4
    FCP = compute_feature_class_probs(tofoo, pcd);
end
if nargin < 5
    pmin = 0;
end
if nargin < 6
    num_samples = 50;  %TODO: tune this parameter
end

lambda = .3;  %TODO: tune this parameter
hard_assignment = 1;

k = length(tofoo.W);
n = size(pcd.X,1);

% compute max BMM densities (for pruning)
bmm_mode_densities = zeros(1,length(tofoo.BMM));
for i=1:length(tofoo.BMM)
    for j=1:length(tofoo.BMM(i).B)
        bmm_mode_densities(i) = max(bmm_mode_densities(i), 1/tofoo.BMM(i).B(j).F);
    end
end
bmm_mode_log_densities = log(bmm_mode_densities);  %dbug


% prob(q) = e^(x1/b1 + x2/b2 + ... + xk/bk)
PX = zeros(1,k);  % total class log probabilities
PB = zeros(1,k);  % inverse coefficients

%TODO: sample evenly from each feature class?
I = randperm(n);
I = I(1:num_samples);
FCW = sum(FCP(I,:));  % feature class weights for cloud subset
FCW(FCW==0) = inf;

% compute quaternion probabilities
for i=I
    q2 = quaternion_mult(q, pcd.Q(i,:));

    if hard_assignment
        j = find(FCP(i,:));
        p = bingham_mixture_pdf(q2, tofoo.BMM(j).B, tofoo.BMM(j).W);
        PX(j) = PX(j) + log(p);
        PB(j) = PB(j) + 1;
    else
        for j=1:k
            if FCP(i,j) > 0
                p = bingham_mixture_pdf(q2, tofoo.BMM(j).B, tofoo.BMM(j).W);
                PX(j) = PX(j) + log(p)*FCP(i,j);
                PB(j) = PB(j) + FCP(i,j);
            end
        end
    end
    
    % pruning
    log_pmax = lambda*sum((PX + bmm_mode_log_densities.*(1-(PB./FCW)))./FCW);
    if log_pmax < log(pmin)
        p = 0;
        return;
    end
end

PB(PB==0) = inf;
p = exp(lambda*sum(PX./PB));

