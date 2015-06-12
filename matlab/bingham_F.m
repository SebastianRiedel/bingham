function [F dF] = bingham_F(z)
%[F dF] = bingham_F(z) -- looks up 2*1F1(1/2; (d+1)/2; z) and partial
%derivatives w.r.t. z

global bingham_constants_
if ~exist('bingham_constants_') || isempty(bingham_constants_)
    fprintf('Loading bingham constants...');
    bingham_constants_ = load_bingham_constants();
    fprintf('done\n');
end
F_cache = bingham_constants_;

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
    X = F_cache.F{d}([zi0(1),zi1(1)])';
elseif d==2
    X = F_cache.F{d}([zi0(1),zi1(1)], [zi0(2),zi1(2)]);
elseif d==3
    X = F_cache.F{d}([zi0(1),zi1(1)], [zi0(2),zi1(2)], [zi0(3),zi1(3)]);
end

F = interp_linear(X,alpha);

if nargout>=2
    if d==1
        X = F_cache.dF{d}(1, [zi0(1),zi1(1)]);
        dF = interp_linear(X', alpha);
    elseif d==2
        X1 = F_cache.dF{d}(1, [zi0(1),zi1(1)], [zi0(2),zi1(2)]);
        X2 = F_cache.dF{d}(2, [zi0(1),zi1(1)], [zi0(2),zi1(2)]);
        dF1 = interp_linear(reshape(X1, [2,2]), alpha);
        dF2 = interp_linear(reshape(X2, [2,2]), alpha);
        dF = [dF1, dF2];
    elseif d==3
        X1 = F_cache.dF{d}(1, [zi0(1),zi1(1)], [zi0(2),zi1(2)], [zi0(3),zi1(3)]);
        X2 = F_cache.dF{d}(2, [zi0(1),zi1(1)], [zi0(2),zi1(2)], [zi0(3),zi1(3)]);
        X3 = F_cache.dF{d}(3, [zi0(1),zi1(1)], [zi0(2),zi1(2)], [zi0(3),zi1(3)]);
        dF1 = interp_linear(reshape(X1, [2,2,2]), alpha);
        dF2 = interp_linear(reshape(X2, [2,2,2]), alpha);
        dF3 = interp_linear(reshape(X3, [2,2,2]), alpha);
        dF = [dF1, dF2, dF3];
    end
end


