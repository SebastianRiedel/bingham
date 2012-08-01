function pcd = load_pcd(filename)
% pcd = load_pcd(filename)

f = fopen(filename);

columns = {};
vp = [];
width = [];
height = [];

% read header
while 1
   s = fgets(f);
   [t s] = strtok(s);
   if strcmp(t, 'COLUMNS') || strcmp(t, 'FIELDS')
       
      % convert new fields names to old field names
      s = strrep(s, 'normal_x', 'nx');
      s = strrep(s, 'normal_y', 'ny');
      s = strrep(s, 'normal_z', 'nz');
      s = strrep(s, 'principal_curvature_x', 'pcx');
      s = strrep(s, 'principal_curvature_y', 'pcy');
      s = strrep(s, 'principal_curvature_z', 'pcz');
      s = strrep(s, 'fpfh', ['f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 ' ...
                             'f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f30 f31 f32 f33']);
      
      i = 0;  %length(columns);
      s = strtrim(s);
      while ~isempty(s)
         i = i+1;
         [t s] = strtok(s);
         columns{i} = t;
      end
   elseif strcmp(t, 'VIEWPOINT')
       vp = sscanf(s, '%f', [1 inf]);
   elseif strcmp(t, 'WIDTH')
       width = sscanf(s, '%f');
   elseif strcmp(t, 'HEIGHT')
       height = sscanf(s, '%f');
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

pcd = populate_pcd_fields(columns, data);
pcd.vp = vp;
if ~isempty(width) && ~isempty(height)
    pcd.width = width;
    pcd.height = height;
else
    pcd.width = size(data,1);
    pcd.height = 1;
end

