function pcd = populate_pcd_fields(columns, data)
% pcd = populate_pcd_fields(columns, data) -- populate the fields of a pcd

pcd.columns = columns;
pcd.data = data;

ch_cluster = find(strcmp(columns, 'cluster'));
ch_x = find(strcmp(columns, 'x'));
ch_y = find(strcmp(columns, 'y'));
ch_z = find(strcmp(columns, 'z'));
ch_pfh = find(strncmp(columns, 'f', 1));
ch_nx = find(strcmp(columns, 'nx'));
ch_ny = find(strcmp(columns, 'ny'));
ch_nz = find(strcmp(columns, 'nz'));
ch_curv = find(strcmp(columns, 'curvature'));
ch_pcx = find(strcmp(columns, 'pcx'));
ch_pcy = find(strcmp(columns, 'pcy'));
ch_pcz = find(strcmp(columns, 'pcz'));
ch_pc1 = find(strcmp(columns, 'pc1'));
ch_pc2 = find(strcmp(columns, 'pc2'));
ch_red = find(strcmp(columns, 'red'));
ch_green = find(strcmp(columns, 'green'));
ch_blue = find(strcmp(columns, 'blue'));
ch_balls = find(strcmp(columns, 'balls'));
ch_segments = find(strcmp(columns, 'segments'));

if ~isempty(ch_cluster)
   pcd.L = data(:, ch_cluster);
   pcd.k = max(pcd.L)+1;
end
if ~isempty(ch_x)
   pcd.X = data(:, ch_x);
end
if ~isempty(ch_y)
   pcd.Y = data(:, ch_y);
end
if ~isempty(ch_z)
   pcd.Z = data(:, ch_z);
end
if ~isempty(ch_pfh)
   pcd.F = data(:, ch_pfh);
end
if ~isempty(ch_nx)
   pcd.NX = data(:, ch_nx);
end
if ~isempty(ch_ny)
   pcd.NY = data(:, ch_ny);
end
if ~isempty(ch_nz)
   pcd.NZ = data(:, ch_nz);
end
if ~isempty(ch_curv)
    pcd.C = data(:, ch_curv);
end
if ~isempty(ch_pcx)
   pcd.PCX = data(:, ch_pcx);
end
if ~isempty(ch_pcy)
   pcd.PCY = data(:, ch_pcy);
end
if ~isempty(ch_pcz)
   pcd.PCZ = data(:, ch_pcz);
end
if ~isempty(ch_pc1)
   pcd.PC1 = data(:, ch_pc1);
end
if ~isempty(ch_pc2)
   pcd.PC2 = data(:, ch_pc2);
end
if ~isempty(ch_nx) && ~isempty(ch_pcx)
   pcd.Q = get_pcd_quaternions(pcd.data, pcd.columns);
end
if ~isempty(ch_pfh) && ~isempty(ch_cluster)
    pcd.M = zeros(pcd.k, size(pcd.F,2));
    pcd.V = zeros(1, pcd.k);
    for i=1:pcd.k
        pcd.M(i,:) = mean(pcd.F(pcd.L==i-1,:));
        pcd.V(i) = sum(var(pcd.F(pcd.L==i-1,:)));
    end
end
if ~isempty(ch_red)
   pcd.R = data(:, ch_red);
end
if ~isempty(ch_green)
   pcd.G = data(:, ch_green);
end
if ~isempty(ch_blue)
   pcd.B = data(:, ch_blue);
end
if ~isempty(ch_balls)
   pcd.balls = data(:, ch_balls);
end
if ~isempty(ch_segments)
   pcd.segments = data(:, ch_segments);
end


