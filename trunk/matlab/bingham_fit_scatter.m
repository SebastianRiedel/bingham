function B = bingham_fit_scatter(S, nn_lookup)

global bingham_constants_
if ~exist('bingham_constants_') || isempty(bingham_constants_)
    fprintf('Loading bingham constants...');
    bingham_constants_ = load_bingham_constants();
    fprintf('done\n');
end

if nargin < 2
    nn_lookup = 0;
end

d = size(S,1);

[V,D] = eig(S);
evals = diag(D);
V = V(:,1:d-1);

dY = evals(1:end-1)';

n = size(bingham_constants_.dY{d-1}, 1);
[~,i] = min(sum((bingham_constants_.dY{d-1} - repmat(dY, [n,1])).^2, 2));
nn_idx = bingham_constants_.dY_indices{d-1}(i,:);

B.Z = -bingham_constants_.Z(nn_idx);

if ~nn_lookup
    B.Z = fmincon(@(z) dY_cost_fn(z,dY), B.Z, [], [], [], [], -bingham_constants_.Z(end)*ones(1,d-1), zeros(1,d-1));
end


B.d = d;
B.V = V;
[B.F B.dF] = bingham_F(B.Z);

end


function g = dY_cost_fn(Z,dY)
    [F dF] = bingham_F(Z);
    g = 1000000*sqrt(sum((dF/F - dY).^2));
    
    if isnan(g)
        g = Inf;
    end
    
    %Z
    %g
end
