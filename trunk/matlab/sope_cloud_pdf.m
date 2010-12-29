function p = sope_cloud_pdf(x, q, tofoo, pcd, FCP, num_samples, lambda)
%p = sope_cloud_pdf(q, tofoo, pcd, FCP, num_samples, lambda) -- computes the
%pdf of a point cloud given a sope model (tofoo) and a model pose, (x,q).


if nargin < 6 || isempty(num_samples)
    num_samples = 10;
end
if nargin < 7 || isempty(lambda)
    lambda = 4; %.07; %.3;
end

k = length(tofoo.W);
n = size(pcd.X, 1);
hard_assignment = 1;

%TODO: sample evenly from each feature class?
I = randperm(n);
I = I(1:num_samples);

if nargin < 5 || isempty(FCP)
    FCP = compute_feature_class_probs(tofoo, pcd, hard_assignment, I);
end

% prob(q) = lambda*e^(lambda*(p1/b1 + p2/b2 + ... + pk/bk))
P = zeros(1,k);  % total class log probabilities
%PB = zeros(1,k);  % inverse coefficients
PB = repmat(num_samples, [1,k]);

q_inv = [q(1) -q(2:4)];

% compute quaternion probabilities
for i=I
    if rand() < .5
        q2 = quaternion_mult(pcd.Q(i,:,1), q_inv);
    else
        q2 = quaternion_mult(pcd.Q(i,:,2), q_inv);
    end

    xi = [pcd.X(i); pcd.Y(i); pcd.Z(i)] - x';
    R_inv = quaternionToRotationMatrix(q_inv);
    x2 = (R_inv*xi)';
    
    if hard_assignment
        j = find(FCP(i,:));
        
        % p(q2)
        p = bingham_mixture_pdf(q2, tofoo.BMM(j).B, tofoo.BMM(j).W);
        P(j) = P(j) + log(p);
        
        % p(x2|q2)
        [x_mean x_cov] = qksample_tofoo(q2,j,tofoo);
        p = mvnpdf(x2, x_mean, x_cov);
        P(j) = P(j) + log(p);
    else
        fprintf('soft assignment not supported\n');
        return
    end
end

p = lambda*exp(lambda*sum(P./PB));

