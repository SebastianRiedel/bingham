function FCP = compute_feature_class_probs(tofoo, pcd, hard_assignment)
%FCP = compute_feature_class_probs(tofoo, pcd)

if nargin < 3
    hard_assignment = 1;
end

k = length(tofoo.W);
n = size(pcd.X,1);

FCP = zeros(n,k);  % feature class probabilities

for i=1:n
    % compute feature class probabilities
    DF = tofoo.M - repmat(pcd.F(i,:), [k,1]);
    df = sum(DF.*DF,2);
    %feature_class_prob = zeros(k,1);
    %for j=1:k  % for each feature type
    %    feature_class_prob(j) = tofoo.W(j) * normpdf(df(j), 0, 1000);  %tofoo.V(j)
    %end
    feature_class_prob = tofoo.W .* normpdf(df, 0, 1000);
    feature_class_prob = feature_class_prob/sum(feature_class_prob);

    if hard_assignment
        feature_class_prob = feature_class_prob .* (feature_class_prob==max(feature_class_prob));
        feature_class_prob = feature_class_prob/sum(feature_class_prob);
    end
    
    FCP(i,:) = feature_class_prob';
end
