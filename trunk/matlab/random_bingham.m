function [V Z F] = random_bingham(d, z0)
% [V Z F] = random_bingham(d, z0)


V = zeros(d, d-1);

V(:,1) = rand(d,1);