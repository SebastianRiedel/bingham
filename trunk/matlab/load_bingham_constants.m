function bingham_constants = load_bingham_constants()

data = load('bingham_constants.mat');
bingham_constants = data.bingham_constants;


%%%  add symmetries in F-lookup tables  %%%
% 3-D
for i=1:length(bingham_constants.Z)
    for j=1:i
        for k=1:j
            F = bingham_constants.F{3}(i,j,k);
            bingham_constants.F{3}(i,k,j) = F;
            bingham_constants.F{3}(j,i,k) = F;
            bingham_constants.F{3}(j,k,i) = F;
            bingham_constants.F{3}(k,i,j) = F;
            bingham_constants.F{3}(k,j,i) = F;
        end
    end
end
% 2-D
for i=1:length(bingham_constants.Z)
    for j=1:i
        F = bingham_constants.F{2}(i,j);
        bingham_constants.F{2}(j,i) = F;
    end
end


%%%  add symmetries in dF-lookup tables  %%%
% 3-D
perms = [1,3,2; 2,1,3; 2,3,1; 3,1,2; 3,2,1];
for i=1:length(bingham_constants.Z)
    for j=1:i
        for k=1:j
            idx = [i,j,k];
            dF = bingham_constants.dF{3}(:,i,j,k);
            for d=1:3
                for p=1:size(perms,1)
                    bingham_constants.dF{3}(d, idx(perms(p,1)), idx(perms(p,2)), idx(perms(p,3))) = dF(perms(p,d));
                end
            end
        end
    end
end
% 2-D
for i=1:length(bingham_constants.Z)
    for j=1:i
        dF = bingham_constants.dF{2}(:,i,j);
        bingham_constants.dF{2}(1,j,i) = dF(2);
        bingham_constants.dF{2}(2,j,i) = dF(1);
    end
end


%%%  compute dY = dF/F lookup tables  %%%
% 1-D
n = length(bingham_constants.Z);
bingham_constants.dY{1} = (bingham_constants.dF{1} ./ bingham_constants.F{1});
bingham_constants.dY_indices{1} = 1:n;

% 2-D
dY = zeros(n^2, 2);
dY_indices = zeros(n^2, 2);
cnt=0;
for i=1:n
    for j=1:i
        cnt=cnt+1;
        dY(cnt,:) = bingham_constants.dF{2}(:,i,j)' / bingham_constants.F{2}(i,j);
        dY_indices(cnt,:) = [i,j];
    end
end
bingham_constants.dY{2} = dY(1:cnt, :);
bingham_constants.dY_indices{2} = dY_indices(1:cnt, :);

% 3-D
dY = zeros(n^3, 3);
dY_indices = zeros(n^3, 3);
cnt=0;
for i=1:n
    for j=1:i
        for k=1:j
            cnt=cnt+1;
            dY(cnt,:) = bingham_constants.dF{3}(:,i,j,k)' / bingham_constants.F{3}(i,j,k);
            dY_indices(cnt,:) = [i,j,k];
        end
    end
end
bingham_constants.dY{3} = dY(1:cnt, :);
bingham_constants.dY_indices{3} = dY_indices(1:cnt, :);


