function BMM = tofoo_posterior(tofoo, q, f)
%BMM = tofoo_posterior(tofoo, q, f) -- computes the Bingham mixture
%resulting from the posterior distribution on object orientation given an
%observed quaternion, q, and feature, f.

hard_assignment = 1;

k = length(tofoo.W);

% compute feature class probabilities
DF = tofoo.M - repmat(f, [k,1]);
df = sum(DF.*DF,2);
feature_class_prob = zeros(k,1);
for i=1:k  % for each feature type
    feature_class_prob(i) = tofoo.W(i) * normpdf(df(i), 0, 1000);  %tofoo.V(i)  %TODO: tune this parameter
end
feature_class_prob = feature_class_prob/sum(feature_class_prob);

if hard_assignment
    feature_class_prob = feature_class_prob .* (feature_class_prob==max(feature_class_prob));
    feature_class_prob = feature_class_prob/sum(feature_class_prob);
end

n = 0;
%BMM.B = struct();
BMM.W = [];
for i=1:k
    if feature_class_prob(i) > 0
        for j=1:length(tofoo.BMM(i).B)
            % invert and rotate
            B = tofoo.BMM(i).B(j);
            W = tofoo.BMM(i).W(j);
            B = bingham_invert_3d(B);
            B = bingham_rotate_3d(B,q);
            
            % add to the mixture
            n = n+1;
            BMM.B(n) = B;
            BMM.W(n) = W * feature_class_prob(i);
        end
    end
end


