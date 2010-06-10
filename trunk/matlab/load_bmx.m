function [B,W] = load_bmx(bmx_file)
% [B,W] = load_bmx(bmx_file) -- loads a 3D BMX file

f = fopen(bmx_file, 'r');

X = fscanf(f, 'B %d %d %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n', [23 inf]);

B = {};
W = {};
for r=1:size(X,2)  % for each row in the file
    c = X(1,r) + 1;                           % cluster
    i = X(2,r) + 1;                           % mixture component
    W{c}(i) = X(3,r);                         % mixture weight
    B{c}(i).d = X(4,r);                       % dimensions
    B{c}(i).F = X(5,r);                       % normalization constant
    B{c}(i).dF = X(6:8,r);                    % concentration params
    B{c}(i).Z = X(9:11,r);                    % concentration params
    B{c}(i).V = reshape(X(12:23,r), [4 3]);   % orthogonal direction vectors
end

fclose(f);
