function F = bingham_1F1(z, F_cache)
%F = bingham_1F1(z, F_cache) -- computes 2*1F1(1/2; (d+1)/2; z)


z = sort(z);

if z(end) > 0
    fprintf('Error: z > 0\n');
elseif -z(1) < eps
    F = surface_area_hypersphere(length(z));
elseif -z(end) < eps
    F = sqrt(pi) * bingham_1F1(z(1:end-1), F_cache);
else
    
    z = [0, z(end:-1:2)] - z(1)
    d = length(z);

    zi = zeros(1,d);
    for i=1:d
        zi(i) = find(F_cache.Z == z(i), 1);
    end
    if d==1
        F = F_cache.table{d}(zi(1));
    elseif d==2
        F = F_cache.table{d}(zi(1), zi(2));
    elseif d==3
        F = F_cache.table{d}(zi(1), zi(2), zi(3));
    end
end