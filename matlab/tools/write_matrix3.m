function write_matrix3(X, filename)
% write_matrix3(X, filename)

f = fopen(filename, 'w');
fprintf(f, '%d %d %d\n', size(X,1), size(X,2), size(X,3));
for i=1:size(X,1)
    for j=1:size(X,2)
        for k=1:size(X,3)
            fprintf(f, '%f ', X(i,j,k));
        end
        fprintf(f, '\n');
    end
end
fclose(f);
