function grid2 = smooth_occ_grid(grid)
% grid2 = smooth_occ_grid(grid)

grid2 = grid;

occ = zeros(size(grid.occ)+2);
occ(2:end-1, 2:end-1, 2:end-1) = grid.occ;

for x=1:size(grid.occ,1)
    for y=1:size(grid.occ,2)
        for z=1:size(grid.occ,3)
            v = occ(x+1,y+1,z+1) + .25*(occ(x,y+1,z+1) + occ(x+1,y,z+1) + ...
                occ(x+1,y+1,z) + occ(x+2,y+1,z+1) + occ(x+1,y+2,z+1) + ...
                occ(x+1,y+1,z+2));
            grid2.occ(x,y,z) = v/2.5;
        end
    end
end
