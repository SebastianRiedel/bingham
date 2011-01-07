function [H,W] = mope_sample(models, pcd, occ_grid, n, k, FCP)
%[H,W] = mope_sample(models, pcd, occ_grid, n, k) -- sample 'n' scene hypotheses, 'H',
%with 'k' objects per sample, given models 'models' and a point cloud, 'pcd'
%with occupancy grid, 'occ_grid'

if nargin < 5 || isempty(FCP)
    for m=1:length(models)
        fprintf('.');
        FCP{m} = compute_feature_class_probs(models(m).tofoo, pcd, 1);
    end
    fprintf('\n');
end

nh = n; %10000;  %dbug
H = cell(1,nh);   % object hypotheses
W = zeros(1,nh);  % hypothesis weights
npoints = size(pcd.X,1);

for i=1:nh
    
    % 1. sample pcd point
    j = 3242; %ceil(rand()*npoints);  %dbug
    f = pcd.F(j,:);
    if rand() < .5
        q = pcd.Q(j,1:4);
    else
        q = pcd.Q(j,5:8);
    end

    % 2. (optional) filter point samples
    
    % 3. sample object id given point feature
    pfh_probs = zeros(1,length(models));
    for m=1:length(models)
        pfh_probs(m) = tofoo_feature_pdf(f, models(m).tofoo);
    end
    pfh_probs = pfh_probs/sum(pfh_probs);
    m = 1; %pmf_sample(pfh_probs);  %dbug
    tofoo = models(m).tofoo;
    H{i}.id = m;
    
    % 4. sample object pose given point and id
    % compute the model orientation posterior given the feature
    BMM = tofoo_posterior(tofoo, q, f);
    % sample an orientation from the proposal distribution
    r2 = bingham_mixture_sample(BMM.B, BMM.W, 1);
    p2 = bingham_mixture_pdf(r2, BMM.B, BMM.W);
    % sample from the proposal distribution of position given orientation
    xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];
    c = find(FCP{m}(j,:));
    q2 = quaternion_mult(q, qinv(r2));
    [x_mean x_cov] = qksample_tofoo(q2,c,tofoo);
    x0 = mvnrnd(x_mean, x_cov);
    R = quaternionToRotationMatrix(r2);
    x2 = (xj' - R*x0')';
    p2 = p2*mvnpdf(x0, x_mean, x_cov);

    H{i}.x = x2;
    H{i}.q = r2;
    
    % 5. weight object hypothesis using occupancy grid
    occ_logp = 0;
    model_pcd = models(H{i}.id).pcd;
    for gi=1:500
        gj = ceil(rand()*size(model_pcd.X, 1));
        p = [pcd.X(gj) pcd.Y(gj) pcd.Z(gj)];          % observed point
        c = ceil((p - occ_grid.min)/occ_grid.res);    % point cell
        if (c >= [1,1,1]) .* (c <= size(occ_grid.occ))
            logp = log(occ_grid.occ(c(1), c(2), c(3)));
        else
            logp = log(.5);
        end
        occ_logp = occ_logp + logp;
    end
    t2 = exp(occ_logp);
    W(i) = t2/p2;
    
    %t2 = mope_hypothesis_pdf(h2, models(h2.id), pcd, 10, FCP{h2.id});
    
    fprintf('.');
end
fprintf('\n');


[W I] = sort(W,'descend');
H2 = cell(1,nh);
for i=1:nh
    H2{i} = H{I(i)};
end
H = H2;
W = W/sum(W);


