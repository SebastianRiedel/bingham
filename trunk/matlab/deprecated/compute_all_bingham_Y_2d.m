function Y = compute_all_bingham_Y_2d(r,iter)
%Y = compute_all_bingham_Y_2d(n,iter)

n = length(r);
Y = zeros(n,n,2);
for i=1:n
    fprintf('.');
    for j=1:n
        Y(i,j,:) = reshape(bingham_Y_2d(r(i),r(j),iter), [1 1 2]);
    end
end
fprintf('\n');
