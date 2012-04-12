function F = bingham_F(z, F_cache)
%F = bingham_1F1(z, F_cache) -- looks up 2*1F1(1/2; (d+1)/2; z)


z = -z;

d = length(z);

zi0 = zeros(1,d);
zi1 = zeros(1,d);
alpha = zeros(1,d);
for i=1:d
    j = find(F_cache.Z >= z(i), 1);
    if isempty(j)
        zi0(i) = length(F_cache.Z);
        zi1(i) = length(F_cache.Z);
        alpha(i) = 0;
    elseif j==1
        zi0(i) = 1;
        zi1(i) = 1;
        alpha(i) = 0;
    else
        zi0(i) = j-1;
        zi1(i) = j;
        alpha(i) = (z(i) - F_cache.Z(j-1)) / (F_cache.Z(j) - F_cache.Z(j-1));
    end
end

if d==1
    X = F_cache.table{d}([zi0(1),zi1(1)]);
elseif d==2
    X = F_cache.table{d}([zi0(1),zi1(1)], [zi0(2),zi1(2)]);
elseif d==3
    X = F_cache.table{d}([zi0(1),zi1(1)], [zi0(2),zi1(2)], [zi0(3),zi1(3)]);
end

F = interp_linear(X,alpha);
