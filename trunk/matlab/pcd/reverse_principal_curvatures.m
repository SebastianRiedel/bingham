function q2 = reverse_principal_curvatures(q)

%R = quaternion_to_rotation_matrix(q);
%q2 = rotation_matrix_to_quaternion([R(:,1), -R(:,2), -R(:,3)])

q2 = [-q(2), q(1), q(4), -q(3)];

if size(q,1) == 4
    q2 = q2';
end
