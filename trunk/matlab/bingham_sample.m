function X = bingham_sample(B,n)
% X = bingham_sample(B,n) - sample n points from a Bingham using Monte Carlo simulation

burn_in = 5; %10;
sample_rate = 1; %10;
%sigma = .1;

x = bingham_mode(B);
S = bingham_scatter(B);
d = length(x);
z = zeros(1,d);
t = bingham_pdf(x,B);  % target
p = mvnpdf(x,z,S);    % proposal

num_accepts = 0;
for i=1:n*sample_rate+burn_in
    %input(':')
    %f = bingham_pdf(x,B)
    %x2 = normrnd(x, sigma);
    x2 = mvnrnd(z,S);
    x2 = x2/norm(x2);
    t2 = bingham_pdf(x2,B);
    p2 = mvnpdf(x2,z,S);
    a1 = t2 / t;
    a2 = p / p2;
    a = a1*a2;
    if a > rand()
        x = x2;
        p = p2;
        t = t2;
        num_accepts = num_accepts + 1;
    end
    %if x(1) < 0
    %    x = -x;
    %end
    X(i,:) = x;
end

%accept_rate = num_accepts / (n*sample_rate + burn_in)

X = X(burn_in+1:sample_rate:end,:);
