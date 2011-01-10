function I = pcd_random_walk(pcd, i0, n, sigma)
%I = pcd_random_walk(pcd, i0, n, sigma)

I(1) = i0;
i = i0;
for cnt=2:n
    x = [pcd.X(i), pcd.Y(i), pcd.Z(i)];
    x2 = normrnd(x,sigma);
    D2 = (pcd.X - x2(1)).^2 + (pcd.Y - x2(2)).^2 + (pcd.Z - x2(3)).^2;
    D2(i) = inf;
    [dmin i] = min(D2);
    I(cnt) = i;
end
