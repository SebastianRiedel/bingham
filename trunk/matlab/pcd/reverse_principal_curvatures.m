function q2 = reverse_principal_curvatures(q)

q2 = [-q(2), q(1), q(4), -q(3)];

if size(q,1) == 4
    q2 = q2';
end
