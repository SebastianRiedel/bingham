function w = orthogonal_vector(v)
% w = orthogonal_vector(v) -- finds a unit vector orthogonal to v.

i = find(v,1);

w = 0*v;
if isempty(i)  % v contains all zeros
    w(1) = 1;
elseif length(v)==1  % v is a non-zero scalar
    w = 0;
else
    i2 = mod(i,length(v))+1;
    w(i) = v(i2);
    w(i2) = -v(i);
    w = w/norm([w(i), w(i2)]);
end
