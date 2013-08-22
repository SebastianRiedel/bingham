function export_bingham_constants(bingham_constants)
% save to a .h file

fname = 'bingham_constant_tables.h';
f = fopen(fname, 'w');

N = length(bingham_constants.Z);

% write header
fprintf(f, '#ifndef BINGHAM_CONSTANT_TABLES\n');
fprintf(f, '#define BINGHAM_CONSTANT_TABLES\n\n');
fprintf(f, 'const int BINGHAM_TABLE_LENGTH = %d;\n\n', N);

write_array_1d(f, sqrt(bingham_constants.Z), 'bingham_table_range');
write_array_1d(f, bingham_constants.F{1}, 'bingham_F_table_1d');
write_array_1d(f, bingham_constants.dF{1}, 'bingham_dF_table_1d');

write_array_2d(f, bingham_constants.F{2}, 'bingham_F_table_2d');
write_array_2d(f, reshape(bingham_constants.dF{2}(1,:,:), [N,N]), 'bingham_dF1_table_2d');
write_array_2d(f, reshape(bingham_constants.dF{2}(2,:,:), [N,N]), 'bingham_dF2_table_2d');

write_array_3d(f, bingham_constants.F{3}, 'bingham_F_table_3d');
write_array_3d(f, reshape(bingham_constants.dF{3}(1,:,:,:), [N,N,N]), 'bingham_dF1_table_3d');
write_array_3d(f, reshape(bingham_constants.dF{3}(2,:,:,:), [N,N,N]), 'bingham_dF2_table_3d');
write_array_3d(f, reshape(bingham_constants.dF{3}(3,:,:,:), [N,N,N]), 'bingham_dF3_table_3d');

% write footer
fprintf(f, '#endif\n');

fclose(f);

end


%==========================================%


function write_array(f,A)
    N = length(A);
    fprintf(f, '{');
    for i=1:N-1
        fprintf(f, '%e, ', A(i));
    end
    fprintf(f, '%e}', A(N));
end

function write_array_1d(f,A,name)
    N = length(A);
    fprintf(f, 'const double %s[%d] = ', name, N);
    write_array(f, A);
    fprintf(f, ';\n\n');
end

function write_array_2d(f,A,name)
    N = size(A,1);
    M = size(A,2);
    fprintf(f, 'const double %s[%d][%d] = {\n', name, N, M);
    for i=1:N
        write_array(f, A(i,:));
        if i<N, fprintf(f, ','); end
        fprintf(f, '\n');
    end
    fprintf(f, '};\n\n');
end

function write_array_3d(f,A,name)
    N = size(A,1);
    M = size(A,2);
    K = size(A,3);
    fprintf(f, 'const double %s[%d][%d][%d] = {\n', name, N, M, K);
    for i=1:N
        fprintf(f, '{\n');
        for j=1:M
            write_array(f, A(i,j,:));
            if j<M, fprintf(f, ','); end
            fprintf(f, '\n');
        end
        fprintf(f, '}');
        if i<N, fprintf(f, ','); end
        fprintf(f, '\n');
    end
    fprintf(f, '};\n\n');
end


