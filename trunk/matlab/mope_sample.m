function [H,W] = mope_sample(models, pcd, occ_grid, n, k, FCP, obj_id, fclass, pcd_obj_mask)
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
[xxx FCP_max] = max(FCP{obj_id},[],2);
if nargin >= 8 && fclass~=0
    FCP_mask = (FCP_max==fclass);
else
    FCP_mask = ones(size(FCP_max));
end
%FCP_ind = find(FCP_mask);  % feature class filter

% figure(1);
% plot_pcd(pcd);
% hold on;
% plot_pcd(populate_pcd_fields(pcd.columns, pcd.data(FCP_ind,:)), 'c.');
% hold off;

if nargin < 9
    pcd_obj_mask = ones(1,length(pcd.X));
end

pcd_good_points = find(FCP_mask .* pcd_obj_mask);
if isempty(pcd_good_points)
    H = {};
    W = [];
    return
end

nh = n; %10000;  %dbug
H = cell(1,nh);   % object hypotheses
W = zeros(1,nh);  % hypothesis weights
npoints = size(pcd.X,1);

for i=1:nh
    
    % 1. sample pcd point
    j0 = ceil(rand()*length(pcd_good_points));
    j = pcd_good_points(j0);
    xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];

    f = pcd.F(j,:);
    if rand() < .5
        q = pcd.Q(j,1:4);
    else
        q = pcd.Q(j,5:8);
    end

    % 2. (optional) filter point samples
    
    % 3. sample object id given point feature
%     pfh_probs = zeros(1,length(models));
%     for m=1:length(models)
%         pfh_probs(m) = tofoo_feature_pdf(f, models(m).tofoo);
%     end
%     pfh_probs = pfh_probs/sum(pfh_probs);
     m = obj_id; %pmf_sample(pfh_probs);  %dbug
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
    x0 = mvnrnd(x_mean, x_cov);  % feature position in model coordinates
    R = quaternionToRotationMatrix(r2);
    x2 = (xj' - R*x0')';
    p2 = p2*mvnpdf(x0, x_mean, x_cov);

    H{i}.x = x2;
    H{i}.q = r2;
    
    nsamples = 3;
    sigma = .03;
    I = pcd_random_walk(pcd, j, nsamples, sigma);
    t2 = sope_cloud_pdf(H{i}.x, H{i}.q, models(H{i}.id).tofoo, pcd, [], nsamples, .5, I);
    
    W(i) = t2; %/p2;

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


