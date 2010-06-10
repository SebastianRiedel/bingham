function pcd = load_pcd(filename)
% pcd = load_pcd(filename)


f = fopen(filename);

columns = {};

% read header
while 1
   s = fgets(f);
   [t s] = strtok(s);
   if strcmp(t, 'COLUMNS') || strcmp(t, 'FIELDS')
      i = 0;
      s = strtrim(s);
      while ~isempty(s)
         i = i+1;
         [t s] = strtok(s);
         columns{i} = t;
      end
   elseif strcmp(t, 'DATA')
      [t s] = strtok(s);
      if ~strcmp(t, 'ascii')
         fprintf('Error: %s is not an ASCII file!\n', filename);
         fclose(f);
         return;
      end
      break;
   end
end

% read data
data = fscanf(f, '%f', [length(columns) inf])';

fclose(f);

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
ch_pcx = find(strcmp(columns, 'pcx'));
ch_pcy = find(strcmp(columns, 'pcy'));
ch_pcz = find(strcmp(columns, 'pcz'));

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
if ~isempty(ch_pcx)
   pcd.PCX = data(:, ch_pcx);
end
if ~isempty(ch_pcy)
   pcd.PCY = data(:, ch_pcy);
end
if ~isempty(ch_pcz)
   pcd.PCZ = data(:, ch_pcz);
end
if ~isempty(ch_nx) && ~isempty(ch_pcx)
   pcd.Q = get_pcd_quaternions(pcd.data, pcd.columns);
end
if ~isempty(ch_pfh) && ~isempty(ch_cluster)
    pcd.M = zeros(pcd.k, size(pcd.F,2));
    for i=1:pcd.k
        pcd.M(i,:) = mean(pcd.F(pcd.L==i-1,:));
    end
end

