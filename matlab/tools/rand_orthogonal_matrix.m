function V = rand_orthogonal_matrix(n)
%V = rand_orthogonal_matrix(n)

V = randn(n,1);
V = V/norm(V);
for i=2:n
    v = (eye(n) - V*V')*randn(n,1);
    V(:,i) = v/norm(v);
end
