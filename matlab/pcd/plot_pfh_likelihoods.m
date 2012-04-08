function plot_pfh_likelihoods(M, F, lambda)
% plot_pfh_likelihoods(M, F, lambda)

k = size(M,1);

%cmap = gray;

figure(20);
bar(F);
p = zeros(1,k);
for c=1:k
   p(c) = lambda * exp(-lambda * norm(M(c,:) - F));
end
%p = 65 - ceil(64*(p./max(p)));
p = p./max(p);
figure(21);
for c=1:k
   subplot(k,1,c);
   %bar(M(c,:), 'FaceColor', cmap(p(c),:));
   bar(M(c,:), 'FaceColor', 1-[p(c) p(c) p(c)]);
end

