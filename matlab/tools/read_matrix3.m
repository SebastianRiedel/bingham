function X = read_matrix3(fname)
%X = read_matrix3(fname)

f = fopen(fname);
dim = fscanf(f, '%d', 3);
X = fscanf(f, '%f %f %f %f', [dim(3) inf])';
X = permute(reshape(X, [dim(2), dim(1), dim(3)]), [2 1 3]);
fclose(f);
