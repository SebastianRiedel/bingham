function A = affinity_graph(pcd)
%A = affinity_graph(pcd) -- returns A.neighbors, A.weights

lambda = .5;
n = size(pcd.X,1);
X = [pcd.X, pcd.Y, pcd.Z];

for i=1:n
    x = X(i,:);
    
    % find x's nearest neighbors in the 8 octants
    X0 = (X(:,1) < x(1));
    X1 = (X(:,1) > x(1));
    Y0 = (X(:,2) < x(2));
    Y1 = (X(:,2) > x(2));
    Z0 = (X(:,3) < x(3));
    Z1 = (X(:,3) > x(3));
    X000 = X(find(X0.*Y0.*Z0),:);
    X001 = X(find(X0.*Y0.*Z1),:);
    X010 = X(find(X0.*Y1.*Z0),:);
    X011 = X(find(X0.*Y1.*Z1),:);
    X100 = X(find(X1.*Y0.*Z0),:);
    X101 = X(find(X1.*Y0.*Z1),:);
    X110 = X(find(X1.*Y1.*Z0),:);
    X111 = X(find(X1.*Y1.*Z1),:);

    DX = repmat(x, [size(X000,1), 1]) - X000;
    D = sum(DX.*DX, 2);
    [d j] = min(D);
    A.neighbors(i,1) = j;
    A.weights(i,1) = lambda*exp(-lambda*d);

    DX = repmat(x, [size(X001,1), 1]) - X001;
    D = sum(DX.*DX, 2);
    [d j] = min(D);
    A.neighbors(i,2) = j;
    A.weights(i,2) = lambda*exp(-lambda*d);
    
    
end

