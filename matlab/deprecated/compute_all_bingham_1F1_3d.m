function F = compute_all_bingham_1F1_3d(r,iter)
%F = compute_all_bingham_1F1_3d(n,iter)

n = length(r);
F = zeros(n,n,n);
for i=1:n
    for j=1:n
        for k=1:n
            [i j k]
            tic;
            F(i,j,k) = bingham_1F1_3d(r(i),r(j),r(k),iter);
            toc;
        end
    end
end
