function X = read_matrix(fname)
%X = load_matrix(fname)

f = fopen(fname);
dim = fscanf(f, '%d', 2);
X = fscanf(f, '%f %f %f %f', [dim(2) inf])';
fclose(f);
