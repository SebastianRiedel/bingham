function save_pcd(filename, pcd)
% save_pcd(filename, pcd)


f = fopen(filename, 'w');

fprintf(f, '# .PCD v.6 - Point Cloud Data file format\n');

fprintf(f, 'FIELDS ');
for i=1:length(pcd.columns)
    fprintf(f, '%s ', pcd.columns{i});
end
fprintf(f, '\n');

fprintf(f, 'SIZE ');
for i=1:length(pcd.columns)
    fprintf(f, '4 ');
end
fprintf(f, '\n');

fprintf(f, 'TYPE ');
for i=1:length(pcd.columns)
    fprintf(f, 'F ');
end
fprintf(f, '\n');

fprintf(f, 'COUNT ');
for i=1:length(pcd.columns)
    fprintf(f, '1 ');
end
fprintf(f, '\n');

fprintf(f, 'WIDTH %d\n', size(pcd.data, 1));
fprintf(f, 'HEIGHT 1\n');
if isfield(pcd, 'vp') && ~isempty(pcd.vp)
    fprintf(f, 'VIEWPOINT ');
    fprintf(f, '%f ', pcd.vp);
    fprintf(f, '\n');
end
fprintf(f, 'POINTS %d\n', size(pcd.data, 1));
fprintf(f, 'DATA ascii\n');

fmt = repmat('%e ', [1,size(pcd.data,2)]);
fmt = [fmt '\n'];
fprintf(f, fmt, pcd.data');

%for i=1:size(pcd.data,1)
    %for j=1:size(pcd.data,2)
    %    fprintf(f, '%f ', pcd.data(i,j));
    %end
%    fprintf(f, '%f ', pcd.data(i,:));
%    fprintf(f, '\n');
%end

fclose(f);

