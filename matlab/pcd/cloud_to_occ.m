function grid = cloud_to_occ(cloud, res)
% grid = cloud_to_occ(cloud, res) -- convert a point cloud to an
% unnormalized occupancy grid (with point counts in the cells):
% grid = (grid.min, grid.occ, grid.res)

pmin = min(cloud) - res/2;
pmax = max(cloud) + res/2;
padding = (pmax-pmin)/10;
pmin = pmin - padding;
pmax = pmax + padding;

grid.min = pmin;
grid.res = res;
grid.occ = zeros(ceil((pmax-pmin)/res));

for i=1:size(cloud,1)
    c = ceil((cloud(i,:) - grid.min)/grid.res);  % point cell
    grid.occ(c(1),c(2),c(3)) = grid.occ(c(1),c(2),c(3)) + 1;
end
