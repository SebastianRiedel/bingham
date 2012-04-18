function bingham_constants = load_bingham_constants()

data = load('bingham_constants.mat');
bingham_constants = data.bingham_constants;

for i=1:length(bingham_constants.Z)
    for j=1:i
        for k=1:j
            F = bingham_constants.table{3}(i,j,k);
            bingham_constants.table{3}(i,k,j) = F;
            bingham_constants.table{3}(j,i,k) = F;
            bingham_constants.table{3}(j,k,i) = F;
            bingham_constants.table{3}(k,i,j) = F;
            bingham_constants.table{3}(k,j,i) = F;
        end
    end
end

for i=1:length(bingham_constants.Z)
    for j=1:i
        F = bingham_constants.table{2}(i,j);
        bingham_constants.table{2}(j,i) = F;
    end
end

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

for i=1:length(bingham_constants.Z)
    for j=1:i
        dF = bingham_constants.dF{2}(:,i,j);
        bingham_constants.dF{2}(1,j,i) = dF(2);
        bingham_constants.dF{2}(2,j,i) = dF(1);
    end
end

