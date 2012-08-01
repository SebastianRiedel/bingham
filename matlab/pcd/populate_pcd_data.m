function data = populate_pcd_data(pcd)
% data = populate_pcd_data(pcd) -- populate the data of a pcd

columns = pcd.columns;
data = pcd.data;

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
ch_red = find(strcmp(columns, 'red'));
ch_green = find(strcmp(columns, 'green'));
ch_blue = find(strcmp(columns, 'blue'));
ch_balls = find(strcmp(columns, 'balls'));
ch_segments = find(strcmp(columns, 'segments'));

if ~isempty(ch_cluster)
   data(:, ch_cluster) = pcd.L;
end
if ~isempty(ch_x)
   data(:, ch_x) = pcd.X;
end
if ~isempty(ch_y)
   data(:, ch_y) = pcd.Y;
end
if ~isempty(ch_z)
   data(:, ch_z) = pcd.Z;
end
if ~isempty(ch_pfh)
   data(:, ch_pfh) = pcd.F;
end
if ~isempty(ch_nx)
   data(:, ch_nx) = pcd.NX;
end
if ~isempty(ch_ny)
   data(:, ch_ny) = pcd.NY;
end
if ~isempty(ch_nz)
   data(:, ch_nz) = pcd.NZ;
end
if ~isempty(ch_curv)
    data(:, ch_curv) = pcd.C;
end
if ~isempty(ch_pcx)
   data(:, ch_pcx) = pcd.PCX;
end
if ~isempty(ch_pcy)
   data(:, ch_pcy) = pcd.PCY;
end
if ~isempty(ch_pcz)
   data(:, ch_pcz) = pcd.PCZ;
end
if ~isempty(ch_red)
   data(:, ch_red) = pcd.R;
end
if ~isempty(ch_green)
   data(:, ch_green) = pcd.G;
end
if ~isempty(ch_blue)
   data(:, ch_blue) = pcd.B;
end
if ~isempty(ch_balls)
   data(:, ch_balls) = pcd.balls;
end
if ~isempty(ch_segments)
   data(:, ch_segments) = pcd.segments;
end

