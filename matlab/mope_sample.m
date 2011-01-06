function S = mope_sample(models, pcd, n, k)
%S = mope_sample(models, pcd, n, k) -- sample 'n' scene hypotheses, 'S',
%with 'k' objects per sample, given models 'models' and a point cloud, 'pcd'

for m=1:length(models)
    fprintf('.');
    FCP{m} = compute_feature_class_probs(models(m).tofoo, pcd, 1);
end
fprintf('\n');

H = {};  %object hypotheses
nh = 100; %10000;
npoints = size(pcd.X,1);

burn_in = 10;
sample_rate = 1;
always_accept = 0;
num_accepts = 0;
h = [];
for i=0:nh*sample_rate+burn_in-1
    
    % 1. sample pcd point
    j = ceil(rand()*npoints);
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
    m = pmf_sample(pfh_probs);
    tofoo = models(m).tofoo;
    h2.id = m;
    
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
    
    % 5. accept/reject object hypothesis using mope_hypothesis_pdf()
    h2.x = x2;
    h2.q = r2;
    % sample a random acceptance threshold
    if ~always_accept
        if num_accepts==0
            t2min = 0;
        else
            t2min = rand()*t*p2/p;
        end
        % compute target density for the given orientation 
        t2 = mope_hypothesis_pdf(h2, models(h2.id), pcd, 10, FCP{h2.id});
    end
    if always_accept || num_accepts==0 || t2 > t2min
        num_accepts = num_accepts + 1;
        h = h2;
        p = p2;
        if ~always_accept
            t = t2;
        end
    end
    
    if i-burn_in >= 0 && mod(i-burn_in, sample_rate) == 0
        hi = (i-burn_in)/sample_rate + 1;
        H{hi} = h;
        %H2{hi} = h2;
    end
    
    fprintf('.');
end
fprintf('\n');

S = H; %dbug

