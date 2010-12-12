function tofoo = load_tofoo(name)
%tofoo = load_tofoo(name) -- returns:
%  tofoo.BMM -- bingham mixtures
%  tofoo.M -- feature cluster centers
%  tofoo.V -- feature cluster variances
%  tofoo.W -- feature cluster weights

[B,W] = load_bmx(sprintf('%s.bmx', name));
pcd = load_pcd(sprintf('%s.pcd', name));

tofoo.BMM = [];
for i=1:length(B)
    tofoo.BMM(i).B = B{i};
    tofoo.BMM(i).W = W{i};
end

% TODO -- save this stuff in a separate file
k = max(pcd.L)+1;
tofoo.M = zeros(k, size(pcd.F,2));
tofoo.V = zeros(k, 1);
tofoo.W = zeros(k, 1);

for i=1:k
    F = pcd.F(pcd.L==i-1,:);  % features with label i-1
    tofoo.M(i,:) = mean(F);   % mean feature
    DF = F - repmat(tofoo.M(i,:), [size(F,1),1]);
    tofoo.V(i) = mean(sum(DF.*DF,2));  % feature variance
    tofoo.W(i) = size(F,1) / length(pcd.L);  % feature weight
end

