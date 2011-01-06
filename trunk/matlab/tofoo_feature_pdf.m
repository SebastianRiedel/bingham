function p = tofoo_feature_pdf(f, tofoo)
%p = tofoo_feature_pdf(f, tofoo)

k = size(tofoo.M,1);
F = repmat(f, [k,1]);
DF = F - tofoo.M;
D = sqrt(sum(DF.*DF,2));
p = dot(tofoo.W, normpdf(D,0,tofoo.V));
