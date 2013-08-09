function data = populate_pcd_data(pcd)
% data = populate_pcd_data(pcd) -- populate the data of a pcd

columns = pcd.columns;
data = pcd.data;

ch_cluster = find(strcmp(columns, 'cluster'));
ch_x = find(strcmp(columns, 'x'));
ch_y = find(strcmp(columns, 'y'));
ch_z = find(strcmp(columns, 'z'));
ch_vx = find(strcmp(columns, 'vx'));
ch_vy = find(strcmp(columns, 'vy'));
ch_vz = find(strcmp(columns, 'vz'));
ch_pfh = find(strncmp(columns, 'f', 1));
ch_pfh_small = find(strncmp(columns, 'sf', 2));
ch_sift = find(strncmp(columns, 'sift', 4));
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
ch_surfdist = find(strcmp(columns, 'surfdist'));
ch_surfwidth = find(strcmp(columns, 'surfwidth'));
ch_normalvar = find(strcmp(columns, 'normalvar'));

% ch_m = find(strncmp(columns, 'm', 1));
% ch_cov = find(strncmp(columns, 'cov', 3));
% ch_cnt = find(strncmp(columns, 'cnt', 3));


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
if ~isempty(ch_vx)
   data(:, ch_vx) = pcd.VX;
end
if ~isempty(ch_vy)
   data(:, ch_vy) = pcd.VY;
end
if ~isempty(ch_vz)
   data(:, ch_vz) = pcd.VZ;
end
if ~isempty(ch_pfh)
   data(:, ch_pfh) = pcd.F;
end
if ~isempty(ch_pfh_small)
   data(:, ch_pfh_small) = pcd.F_small;
end
if ~isempty(ch_sift)
   data(:, ch_sift) = pcd.SIFT;
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
if ~isempty(ch_surfdist)
   data(:, ch_surfdist) = pcd.surfdist;
end
if ~isempty(ch_surfwidth)
   data(:, ch_surfwidth) = pcd.surfwidth;
end
if ~isempty(ch_normalvar)
   data(:, ch_normalvar) = pcd.normalvar;
end


% if ~isempty(ch_m)
%    data(:, ch_m(1:3)) = pcd.M1;
%    data(:, ch_m(4:6)) = pcd.M2;   
%    data(:, ch_cov(1:6)) = pcd.C1(:, [1, 2, 3, 5, 6, 9]);
%    data(:, ch_cov(7:12)) = pcd.C2(:, [1, 2, 3, 5, 6, 9]);
%    data(:, ch_cnt(1)) = pcd.CNT1;
%    data(:, ch_cnt(2)) = pcd.CNT2;
% end