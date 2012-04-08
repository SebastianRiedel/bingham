function L = get_pcd_labels(data, columns)
% L = get_pcd_labels(data, columns)

ch_cluster = find(strcmp(columns, 'cluster'));

if isempty(ch_cluster)
   fprintf('Error: channel "cluster" not found.');
   return
end

L = data(:, ch_cluster);
