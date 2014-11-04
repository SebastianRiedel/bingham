function labels = greedy_cluster(X, dthresh)
%labels = greedy_cluster(X, dthresh)

n = size(X,1);
labels = zeros(1,n);
num_clusters = 0;

for i=1:n
    x = X(i,:);
    if labels(i)==0
        labels(i) = num_clusters + 1;  %matlab
        num_clusters = num_clusters + 1;
    end
        
    for j=i+1:n
        if labels(j)==0 && norm(x - X(j,:)) < dthresh
            labels(j) = labels(i);
        end
    end
end
