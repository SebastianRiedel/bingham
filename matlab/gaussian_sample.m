function X = gaussian_sample(u,s,n)
% X = gaussian_sample(u,s,n) - sample n points from a Gaussian using Monte Carlo simulation

burn_in = 0;
sample_rate = 1;

sigma = s/5;


x = u;

d = length(x);

num_accepts = 0;
for i=1:n*sample_rate+burn_in
    %input(':')
    f = normpdf(x,u,s)
    x2 = normrnd(x, sigma);
    %x2 = x2/norm(x2);
    a = normpdf(x2,u,s) / normpdf(x,u,s)
    if a >= 1 || rand() < a
        x = x2;
        num_accepts = num_accepts + 1;
    end
    X(i,:) = x;
end

accept_rate = num_accepts / (n*sample_rate + burn_in)

X = X(burn_in+1:sample_rate:end,:);
