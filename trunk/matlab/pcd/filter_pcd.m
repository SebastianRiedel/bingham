function pcd2 = filter_pcd(pcd)
% pcd2 = filter_pcd(pcd) -- remove inf's and nan's from a pcd

columns = pcd.columns;
data = pcd.data;

data = data(sum(~isfinite(data),2)==0, :);

pcd2 = populate_pcd_fields(columns, data);
