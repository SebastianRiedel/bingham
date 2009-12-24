function write_matrix(X, filename)
% write_matrix(X, filename)

f = fopen(filename, 'w');
fprintf(f, '%d %d\n', size(X,1), size(X,2));
for i=1:size(X,1)
    for j=1:size(X,2)
        fprintf(f, '%f ', X(i,j));
    end
    fprintf(f, '\n');
end
fclose(f);
