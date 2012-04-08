function Q = get_pcd_quaternions(data, columns)
% Q = get_pcd_quaternions(data, columns) --> returns an Nx4x2 matrix





ch_nx = find(strcmp(columns, 'nx'));
ch_ny = find(strcmp(columns, 'ny'));
ch_nz = find(strcmp(columns, 'nz'));
ch_pcx = find(strcmp(columns, 'pcx'));
ch_pcy = find(strcmp(columns, 'pcy'));
ch_pcz = find(strcmp(columns, 'pcz'));

if isempty(ch_nx)
   fprintf('Error: channel "nx" not found.');
   return
end
if isempty(ch_ny)
   fprintf('Error: channel "ny" not found.');
   return
end
if isempty(ch_nz)
   fprintf('Error: channel "nz" not found.');
   return
end
if isempty(ch_pcx)
   fprintf('Error: channel "pcx" not found.');
   return
end
if isempty(ch_pcy)
   fprintf('Error: channel "pcy" not found.');
   return
end
if isempty(ch_pcz)
   fprintf('Error: channel "pcz" not found.');
   return
end

NX = data(:, ch_nx);
NY = data(:, ch_ny);
NZ = data(:, ch_nz);
PCX = data(:, ch_pcx);
PCY = data(:, ch_pcy);
PCZ = data(:, ch_pcz);

n = size(data,1);
Q = zeros(n, 4, 2);

for j=1:n
   nx = NX(j); ny = NY(j); nz = NZ(j); 
   pcx = PCX(j); pcy = PCY(j); pcz = PCZ(j);
   pcx2 = ny*pcz - nz*pcy;
   pcy2 = nz*pcx - nx*pcz;
   pcz2 = nx*pcy - ny*pcx;

   R = [nx pcx pcx2 ; ny pcy pcy2 ; nz pcz pcz2];
   q = rotationMatrixToQuaternion(R);

   R2 = [nx -pcx -pcx2 ; ny -pcy -pcy2 ; nz -pcz -pcz2];
   q2 = rotationMatrixToQuaternion(R2);

   Q(j,:,1) = q;
   Q(j,:,2) = q2;
end
