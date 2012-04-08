function F = compute_all_bingham_1F1_2d(r,iter)
%F = compute_all_bingham_1F1_2d(n,iter)

n = length(r);
F = zeros(n,n);
for i=1:n
    fprintf('.');
    for j=1:n
        F(i,j) = bingham_1F1_2d(r(i),r(j),iter);
    end
end
fprintf('\n');
