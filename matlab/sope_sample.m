function XQ = sope_sample(tofoo, pcd, n)
%XQ = sope_sample(tofoo, pcd, n) -- monte carlo (metropolis hasting) sampling
%algorithm for object pose given a tofoo model.

hard_assignment = 1;
burn_in = 10;
sample_rate = 1; %10;
always_accept = 0;

npoints = size(pcd.X,1);
FCP = compute_feature_class_probs(tofoo, pcd, hard_assignment);

r = [];
x = [];
num_accepts = 0;
XQ = zeros(n,7);
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

    % sample from the proposal distribution of position given orientation
    xj = [pcd.X(j) pcd.Y(j) pcd.Z(j)];
    c = find(FCP(j,:));
    q2 = quaternion_mult(q, qinv(r2));
    [x_mean x_cov] = qksample_tofoo(q2,c,tofoo);
    x0 = mvnrnd(x_mean, x_cov);
    R = quaternionToRotationMatrix(r2);
    x2 = (xj' - R*x0')';
    p2 = p2*mvnpdf(x0, x_mean, x_cov);
    
    % sample a random acceptance threshold
    if ~always_accept
        if num_accepts==0
            t2min = 0;
        else
            t2min = rand()*t*p2/p;
        end

        % compute target density for the given orientation 
        t2 = sope_cloud_pdf(x2, r2, tofoo, pcd, FCP);
    end

    % a = p*t2/(p2*t) >= rand()
    % t2 >= rand()*t*p2/p
    
    if always_accept || num_accepts==0 || t2 > t2min
        num_accepts = num_accepts + 1;
        x = x2;
        r = r2;
        p = p2;
        if ~always_accept
            t = t2;
        end
    end
    XQ(i,:) = [x r];
    
    fprintf('.');
end
fprintf('\n');

accept_rate = num_accepts / (n*sample_rate + burn_in)

XQ = XQ(burn_in+1:sample_rate:end,:);



