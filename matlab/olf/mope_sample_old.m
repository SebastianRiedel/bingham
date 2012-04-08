function [H,W] = mope_sample_old(models, pcd, occ_grid, n, k, FCP, obj_id, fclass)
%[H,W] = mope_sample_old(models, pcd, occ_grid, n, k) -- sample 'n' scene hypotheses, 'H',
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
FCP_ind = find(FCP_mask);  % feature class filter

figure(1);
plot_pcd(pcd);
hold on;
plot_pcd(populate_pcd_fields(pcd.columns, pcd.data(FCP_ind,:)), 'c.');
hold off;
%if input(':')==0
%    H = {};
%    W = [];
%    return
%end


if obj_id==1
    pcd_obj_mask = (abs(pcd.X-.8)<.05) .* (abs(pcd.Y-.1)<.05);
elseif obj_id==2
    pcd_obj_mask = (abs(pcd.X-.9)<.05) .* (abs(pcd.Y-.02)<.05) .* (pcd.Z>.8);
elseif obj_id==5
    pcd_obj_mask = (abs(pcd.X-.9)<.05) .* (abs(pcd.Y-.02)<.05) .* (pcd.Z<.8);
else
    fprintf('Error: obj_id > 2\n');
    H = {};
    W = [];
    return
end

pcd_obj = populate_pcd_fields(pcd.columns, pcd.data(find(pcd_obj_mask),:));
%pcd_obj_centered = shift_pcd(pcd_obj, -[.8,.1,.8]);

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
%     while abs(xj(1)-.8)>.05 || abs(xj(2)-.1)>.05
%         j0 = ceil(rand()*length(pcd_good_points));
%         j = pcd_good_points(j0);
%         xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];
%     end    
%     j = ceil(rand()*npoints); %3242; %dbug
%     c = find(FCP{1}(j,:));
%     xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];
%     while abs(xj(1)-.8)>.05 || abs(xj(2)-.1)>.05 || c~=1
%         j = ceil(rand()*npoints); %3242; %dbug
%         c = find(FCP{1}(j,:));
%         xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];
%     end
        
%     figure(1);
%     plot_pcd(pcd);
%     hold on;
%     plot_pcd_point(pcd,j,'p');
%     hold off;
%     view(-41,90);
%     input(':');
    
    
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
    
    % 5. weight object hypothesis using occupancy grid
%     occ_logp = 0;
%     model_pcd = models(H{i}.id).pcd;
%     model_samples = 50;
%     for gi=1:model_samples  %size(model_pcd.X, 1)
%         gj = ceil(rand()*size(model_pcd.X, 1));
%         x0 = [model_pcd.X(gj) model_pcd.Y(gj) model_pcd.Z(gj)];
%         p = H{i}.x + x0*R';
%         c = ceil((p - occ_grid.min)/occ_grid.res);    % point cell
%         if (c >= [1,1,1]) .* (c <= size(occ_grid.occ))
%             logp = log(occ_grid.occ(c(1), c(2), c(3)));
%         else
%             logp = log(.5);
%         end
%         occ_logp = occ_logp + logp;
%     end
%     t2 = 4*exp(4*occ_logp/size(model_pcd.X, 1));
    %t2 = exp(-acos(H{i}.q(1)))
    %q = H{i}.q;
    %t2 = exp(-acos(q(1)^2 - q(2)^2 - q(3)^2 + q(4)^2));

    nsamples = 3;
    sigma = .03;
    I = pcd_random_walk(pcd, j, nsamples, sigma);
    t2 = sope_cloud_pdf(H{i}.x, H{i}.q, models(H{i}.id).tofoo, pcd, [], nsamples, .5, I);

    %I = randperm(size(pcd_obj.X,1));
    %I = [j0 I(1:nsamples-1)];
    %t2 = sope_cloud_pdf(H{i}.x, H{i}.q, models(H{i}.id).tofoo, pcd_obj, [], nsamples, .5, I);
    
    
    %t3 = sope_cloud_pdf(H{i}.x - [.8,.1,.8], H{i}.q, models(H{i}.id).tofoo, pcd_obj_centered, [], 10, 1, I)
    %t2 = sope_cloud_pdf([.8,.1,.8], H{i}.q, models(H{i}.id).tofoo, pcd_obj, [], 10, 1, I)
    %t3 = sope_cloud_pdf([0,0,0], H{i}.q, models(H{i}.id).tofoo, shift_pcd(pcd_obj, -[.8,.1,.8]), [], 10, 1, I);
    %t2=t3;
    %H{i}.x = [.8,.1,.8];
    
    W(i) = t2; %/p2;

    
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


