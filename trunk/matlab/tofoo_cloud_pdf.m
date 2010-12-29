function p = tofoo_cloud_pdf(q, tofoo, pcd, FCP, pmin, num_samples, lambda)
%p = tofoo_cloud_pdf(q, tofoo, pcd, FCP, pmin, num_samples, lambda) -- computes the
%pdf of a point cloud given a tofoo model and a model pose, q, with prob.
%threshold pmin (for pruning).


if nargin < 5 || isempty(pmin)
    pmin = 0;
end
if nargin < 6 || isempty(num_samples)
    num_samples = 10;
end
if nargin < 7 || isempty(lambda)
    lambda = 4; %.07; %.3;
end

k = length(tofoo.W);
n = size(pcd.X,1);
hard_assignment = 1;

%TODO: sample evenly from each feature class?
I = randperm(n);
I = I(1:num_samples);

if nargin < 4 || isempty(FCP)
    FCP = compute_feature_class_probs(tofoo, pcd, hard_assignment, I);
end

% compute max BMM densities (for pruning)
%bmm_mode_densities = zeros(1,length(tofoo.BMM));
%for i=1:length(tofoo.BMM)
%    for j=1:length(tofoo.BMM(i).B)
%        bmm_mode_densities(i) = max(bmm_mode_densities(i), 1/tofoo.BMM(i).B(j).F);
%    end
%end
%bmm_mode_log_densities = log(bmm_mode_densities);  %dbug


% prob(q) = lambda*e^(lambda*(x1/b1 + x2/b2 + ... + xk/bk))
PX = zeros(1,k);  % total class log probabilities
%PB = zeros(1,k);  % inverse coefficients
PB = repmat(num_samples, [1,k]);

FCW = sum(FCP(I,:));  % feature class weights for cloud subset
FCW(FCW==0) = inf;

q_inv = [q(1) -q(2:4)];

% compute quaternion probabilities
for i=I
    q2 = quaternion_mult(pcd.Q(i,:), q_inv);

    if hard_assignment
        j = find(FCP(i,:));
        p = bingham_mixture_pdf(q2, tofoo.BMM(j).B, tofoo.BMM(j).W);
        PX(j) = PX(j) + log(p);
        %PB(j) = PB(j) + 1;
    else
        fprintf('soft assignment not supported\n');
        return
    end
    
    % pruning
    %log_pmax = log(lambda) + lambda*sum((PX + bmm_mode_log_densities.*(1-(PB./FCW)))./FCW);
    %if log_pmax < log(pmin)
    %    p = 0;
    %    return;
    %end
end

PB(PB==0) = inf;
p = lambda*exp(lambda*sum(PX./PB));

