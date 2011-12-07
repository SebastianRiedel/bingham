function pcd = merge_pcd_fields(pcd1, pcd2)
% pcd = merge_pcd_fields(pcd1, pcd2)

[C I1 I2] = intersect(pcd1.columns, pcd2.columns);
I2 = setdiff(1:length(pcd2.columns), I2);

columns = [pcd1.columns, pcd2.columns(I2)];
data = [pcd1.data, pcd2.data(:,I2)];

pcd = populate_pcd_fields(columns, data);

if isfield(pcd1, 'vp')
    pcd.vp = pcd1.vp;
elseif isfield(pcd2, 'vp')
    pcd.vp = pcd2.vp;
end
