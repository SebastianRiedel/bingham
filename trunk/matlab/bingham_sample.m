function X = bingham_sample(B,n)
% X = bingham_sample(B,n) - sample n points from a Bingham using Monte Carlo simulation

X = bingham_sample_nd(B,n);


% %TODO: test for B uniform!
% 
% 
% burn_in = 10;
% sample_rate = 1; %10;
% %sigma = .1;
% 
% x = bingham_mode(B);
% S = bingham_scatter(B);
% d = length(x);
% z = zeros(1,d);
% t = bingham_pdf(x,B);  % target
% p = acgpdf(x,S);    % proposal
% 
% num_accepts = 0;
% for i=1:n*sample_rate+burn_in
%     %input(':')
%     %f = bingham_pdf(x,B)
%     %x2 = normrnd(x, sigma);
%     x2 = acgrnd(S);
%     
%     if norm(x2) > .9 && norm(x2) < 1.1
%         x2 = x2/norm(x2);
%         t2 = bingham_pdf(x2,B);
%         p2 = acgpdf(x2,S);
%         a1 = t2 / t;
%         a2 = p / p2;
%         a = a1*a2;
%         if a > rand()
%             x = x2;
%             p = p2;
%             t = t2;
%             num_accepts = num_accepts + 1;
%         end
%     end
%     %if x(1) < 0
%     %    x = -x;
%     %end
%     X(i,:) = x;
% end
% 
% accept_rate = num_accepts / (n*sample_rate + burn_in)
% 
% X = X(burn_in+1:sample_rate:end,:);
