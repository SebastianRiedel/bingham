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
        for k=1:j
            for d=1:3
                F = bingham_constants.dF{3}(d,i,j,k);
                bingham_constants.dF{3}(d,i,k,j) = F;
                bingham_constants.dF{3}(d,j,i,k) = F;
                bingham_constants.dF{3}(d,j,k,i) = F;
                bingham_constants.dF{3}(d,k,i,j) = F;
                bingham_constants.dF{3}(d,k,j,i) = F;
            end
        end
    end
end

for i=1:length(bingham_constants.Z)
    for j=1:i
        for d=1:2
            F = bingham_constants.dF{2}(d,i,j);
            bingham_constants.dF{2}(d,j,i) = F;
        end
    end
end

for i=1:length(bingham_constants.Z)
    for j=1:i
        F = bingham_constants.table{2}(i,j);
        bingham_constants.table{2}(j,i) = F;
    end
end
