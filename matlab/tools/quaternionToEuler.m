function a = quaternionToEuler(q)
% a = quaternionToEuler(q)

q0 = q(1);
q1 = q(2);
q2 = q(3);
q3 = q(4);

a(1) = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2));
a(2) = arcsin(2*(q0*q2 - q3*q1));
a(3) = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3));
