f = fopen('bingham_constants_table.h', 'w');

n = length(y_range);

fprintf(f, 'const double bingham_table_range[%d] = {', n);
fprintf(f, '%f', y_range(1));
fprintf(f, ', %f', y_range(2:end));
fprintf(f, '};\n');
fprintf(f, '\n\n');

fprintf(f, 'const double bingham_F_table[%d][%d][%d] = ', n, n, n);
write_c_matrix_3d(f, F);
fprintf(f, '\n\n');

fprintf(f, 'const double bingham_dF1_table[%d][%d][%d] = ', n, n, n);
write_c_matrix_3d(f, dF1);
fprintf(f, '\n\n');

fprintf(f, 'const double bingham_dF2_table[%d][%d][%d] = ', n, n, n);
write_c_matrix_3d(f, dF2);
fprintf(f, '\n\n');

fprintf(f, 'const double bingham_dF3_table[%d][%d][%d] = ', n, n, n);
write_c_matrix_3d(f, dF3);
fprintf(f, '\n\n');

fprintf(f, 'const double bingham_dY1_table[%d][%d][%d] = ', n, n, n);
write_c_matrix_3d(f, dY1);
fprintf(f, '\n\n');

fprintf(f, 'const double bingham_dY2_table[%d][%d][%d] = ', n, n, n);
write_c_matrix_3d(f, dY2);
fprintf(f, '\n\n');

fprintf(f, 'const double bingham_dY3_table[%d][%d][%d] = ', n, n, n);
write_c_matrix_3d(f, dY3);
fprintf(f, '\n\n');

fclose(f);
