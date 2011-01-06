function p = mope_hypothesis_pdf(H, models, pcd, num_samples, FCP)
%p = mope_hypothesis_pdf(H, models, pcd, num_samples) -- computes the pdf
%of a scene hypothesis, H{:}, where H{i} has fields 'x', 'q', and 'id',
%and models{i} has fields 'tofoo', 'occ_grid',...

if length(H) > 1
    if nargin < 4
        num_samples = 10;
    end
    logp = 0;
    for i=1:length(H)
        logp = logp + log(mope_hypothesis_pdf(H{i}, models, pcd, num_samples));
    end
    %TODO: add object overlap penalty?
    p = exp(logp);
    return
end

tofoo = models(1).tofoo;
occ_grid = models(1).occ_grid;
k = length(tofoo.W);
n = size(pcd.X, 1);
hard_assignment = 1;
if nargin < 4
    num_samples = 10;
end
lambda = 4;

% sample pcd points w.r.t. occ_grid via rejection sampling
I = randperm(n);
J = [];
occ_probs = [];
i = 0;
Rinv = quaternionToRotationMatrix(qinv(H.q));
while length(J) < num_samples
    % sample from pcd
    i = i+1;
    if i > n
        p = 0;
        return
    end
    x = [pcd.X(I(i)); pcd.Y(I(i)); pcd.Z(I(i))];
    % transform into object coordinates
    x = Rinv*(x - H.x');
    % get occupancy grid cell
    c = ceil((x' - occ_grid.min)/occ_grid.res);
    if (c >= [1,1,1]) .* (c <= size(occ_grid.occ))
        if rand() < occ_grid.occ(c(1), c(2), c(3))
            J = [J, I(i)];
            occ_probs = [occ_probs, occ_grid.occ(c(1), c(2), c(3))];
        end
    end
end
I = J;

if nargin < 5 || isempty(FCP)
    FCP = compute_feature_class_probs(tofoo, pcd, hard_assignment, I);
end

% prob(q) = lambda*e^(lambda*(p1/b1 + p2/b2 + ... + pk/bk))
%P = zeros(1,k);  % total class log probabilities
%PB = zeros(1,k);  % inverse coefficients
%PB = repmat(num_samples, [1,k]);

P = 0;

q = H.q;
q_inv = [q(1) -q(2:4)];

% compute quaternion probabilities
for i=I
    if rand() < .5
        q2 = quaternion_mult(pcd.Q(i,:,1), q_inv);
    else
        q2 = quaternion_mult(pcd.Q(i,:,2), q_inv);
    end

    xi = [pcd.X(i); pcd.Y(i); pcd.Z(i)] - H.x';
    R_inv = quaternionToRotationMatrix(q_inv);
    x2 = (R_inv*xi)';
    
    if hard_assignment
        j = find(FCP(i,:));
        
        % p(q2)
        prob_q = bingham_mixture_pdf(q2, tofoo.BMM(j).B, tofoo.BMM(j).W);
        %P(j) = P(j) + log(p);
        
        % p(x2|q2)
        [x_mean x_cov] = qksample_tofoo(q2,j,tofoo);
        prob_x_given_q = mvnpdf(x2, x_mean, x_cov);
        %P(j) = P(j) + log(p);
        
        prob_xq_given_in = prob_q * prob_x_given_q;
        prob_xq_given_out = 1/surface_area_hypersphere(3);
        prob_in = .1;
        prob_out = 1 - prob_in;
        
        prob_xq = prob_xq_given_in*prob_in + prob_xq_given_out*prob_out;
        P = P + log(prob_xq);
        
    else
        fprintf('soft assignment not supported\n');
        return
    end
end

p = lambda*exp(lambda*P);
%p = lambda*exp(lambda*sum(P./PB));




