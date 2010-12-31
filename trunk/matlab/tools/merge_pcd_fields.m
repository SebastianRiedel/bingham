function pcd = merge_pcd_fields(pcd1, pcd2)
% pcd = merge_pcd_fields(pcd1, pcd2)

columns = [pcd1.columns, pcd2.columns];
data = [pcd1.data, pcd2.data];

pcd = populate_pcd_fields(columns, data);
