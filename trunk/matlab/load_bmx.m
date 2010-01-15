function B = load_bmx(bmx_file)
% B = load_bmx(bmx_file) -- loads a 3D BMX file

f = fopen(bmx_file, 'r');


fscanf(f, 'B %d %d %f %d %f ')