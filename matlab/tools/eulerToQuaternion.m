function q = eulerToQuaternion(a)
% q = eulerToQuaternion(a)

if size(a,1) == 3
    a = a';
end

c1 = cos(a(:,1)/2);
c2 = cos(a(:,2)/2);
c3 = cos(a(:,3)/2);
s1 = sin(a(:,1)/2);
s2 = sin(a(:,2)/2);
s3 = sin(a(:,3)/2);

q = [c1.*c2.*c3 + s1.*s2.*s3, s1.*c2.*c3 - c1.*s2.*s3, c1.*s2.*c3 + s1.*c2.*s3, c1.*c2.*s3 - s1.*s2.*c3];
