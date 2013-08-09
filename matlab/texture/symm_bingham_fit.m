function B = symm_bingham_fit(X, symm_type)
%B = symm_bingham_fit(X, symm_type) -- Fit a symmetric Bingham to a set of unit
%vectors in the rows of X.  symm_type can be one of: 'cubic', ...

if strcmp(symm_type, 'cubic')
    
    global SX;
    
    n = size(X,1);
    d = size(X,2);
    k = size(cubic_symm(X(1,:)), 1);
    SX = zeros(k,d,n);
    for i=1:n
        SX(:,:,i) = cubic_symm(X(i,:));
    end
    
    % fit a Bingham to X to initialize params
    B = bingham_fit(X);
    b = B2b(B);
    
    b(1:3) = [-20,-20,-20];  %dbug
    
    % concentration params
    %z = -20*ones(1,3);
    
    % dual quaternions for V
    %v1 = [1,0,0,0]; %rand(1,4);
    %v1 = v1/norm(v1);
    %v2 = [1,0,0,0]; %rand(1,4);
    %v2 = v2/norm(v2);
    %b = [z v1 v2];
    
    %options = optimset('GradObj','on');
    
    b = fmincon(@sb_cost, b, [],[],[],[],[], [zeros(1,3), Inf*ones(1,8)], @sb_constraints); %, options);
    
    B = b2B(b);
        
else
    fprintf('ERROR: Unsupported symmetry type.\n');
    B = [];
end


end


function Q = quaternion_left_mult_matrix(q)

    a = q(1);
    b = q(2);
    c = q(3);
    d = q(4);

    Q = [ a, -b, -c, -d;  b,  a, -d,  c;  c,  d,  a, -b;  d, -c,  b,  a];
end


function Q = quaternion_right_mult_matrix(q)

    a = q(1);
    b = q(2);
    c = q(3);
    d = q(4);

    Q = [ a, -b, -c, -d;  b,  a,  d, -c;  c, -d,  a,  b;  d,  c, -b,  a];
end


function B = b2B(b)

    B.d = 4;
    B.Z = b(1:3);
    q1 = b(4:7) / norm(b(4:7));
    q2 = b(8:11) / norm(b(8:11));
    V1 = quaternion_left_mult_matrix(q1);
    V2 = quaternion_right_mult_matrix(q2);
    V = V1*V2;
    B.V = V(:,1:3);
    [B.F B.dF] = bingham_F(B.Z);
end


function b = B2b(B)

    V = [B.V, bingham_mode(B)];
    if det(V) < 0
        V(:,4) = -V(:,4);
    end
    
    cq = -.25*(V(1,4) + V(2,3) + V(3,2) + V(4,1));
    dp = -.25*(V(1,4) + V(2,3) - V(3,2) - V(4,1));
    as = -.25*(V(1,4) - V(2,3) + V(3,2) - V(4,1));
    br = .25*(V(1,4) - V(2,3) - V(3,2) + V(4,1));
    bs = -.25*(V(1,3) + V(2,4) + V(3,1) + V(4,2));
    ar = -.25*(V(1,3) + V(2,4) - V(3,1) - V(4,2));
    dq = .25*(V(1,3) - V(2,4) + V(3,1) - V(4,2));
    cp = -.25*(V(1,3) - V(2,4) - V(3,1) + V(4,2));
    dr = -.25*(V(1,2) + V(2,1) + V(3,4) + V(4,3));
    cs = .25*(V(1,2) + V(2,1) - V(3,4) - V(4,3));
    bp = -.25*(V(1,2) - V(2,1) + V(3,4) - V(4,3));
    aq = -.25*(V(1,2) - V(2,1) - V(3,4) + V(4,3));
    ap = .25*(V(1,1) + V(2,2) + V(3,3) + V(4,4));
    bq = -.25*(V(1,1) + V(2,2) - V(3,3) - V(4,4));
    cr = -.25*(V(1,1) - V(2,2) + V(3,3) - V(4,4));
    ds = -.25*(V(1,1) - V(2,2) - V(3,3) + V(4,4));
    
    S = [ap,aq,ar,as; bp,bq,br,bs; cp,cq,cr,cs; dp,dq,dr,ds];
    
    j = find(max(abs(S)) > eps, 1);  % find the first non-zero column of S
    i = find(abs(S(:,j)) > eps, 1);  % find the first non-zero entry in column j
    not_j = setdiff(1:4, j);
    pqrs = [0,0,0,0];
    pqrs(j) = 1;
    pqrs(not_j) = S(i,not_j) / S(i,j);
    v2 = pqrs / norm(pqrs);
    abcd = S(:,j)' / v2(j);
    v1 = abcd / norm(abcd);

    b = [B.Z, v1, v2];
    
    %syms a b c d p q r s real;
    %BV = [B.V, bingham_mode(B)];
    %V = [a -b -c -d; b a -d c; c d a -b; d -c b a] * [p -q -r -s; q p s -r; r -s p q; s r -q p];
    %S = solve(V==BV);
    %b = [B.Z, S.a, S.b, S.c, S.d, S.p, S.q, S.r, S.s];
end


function dVdQ = compute_dVdQ(Q)

    a = Q(1);
    b = Q(2);
    c = Q(3);
    d = Q(4);
    p = Q(5);
    q = Q(6);
    r = Q(7);
    s = Q(8);
    
    dVdQ = zeros(4,8,3);
    
    dVdQ(:,:,1) = [p, -q, -r, -s, a, -b, -c, -d;  q, p, s, -r, b, a, -d, c;  r, -s, p, q, c, d, a, -b;  s, r, -q, p, d, -c, b, a];
    dVdQ(:,:,2) = [-q, -p, s, -r, -b, -a, -d, c;  p, -q, r, s, a, -b, c, d;  -s, -r, -q, p, d, -c, -b, -a;  r, -s, -p, -q, -c, -d, a, -b];
    dVdQ(:,:,3) = [-r, -s, -p, q, -c, d, -a, -b;  s, -r, -q, -p, -d, -c, -b, a;  p, q, -r, s, a, b, -c, d;  -q, p, -s, -r, b, -a, -d, -c];
end


function [f dfdb] = sb_cost(b)

    if nargin == 0
        f = 0;
        return
    end

    global SX

    B = b2B(b);
    
%     f = 0;
%     for i=1:size(X,1)
%         f = f + log(symm_bingham_pdf(X(i,:), B));
%     end

    k = size(SX,1);
    d = size(SX,2);
    n = size(SX,3);
    
    E = zeros(k,n);
    for i=1:n
        E(:,i) = exp((SX(:,:,i)*B.V).^2 * B.Z');
    end
    f = n*log(k*B.F) - sum(log(sum(E)));  % negative log-likelihood
    
    b
    f %= f + (1-b(4:7)*b(4:7)')^2 + (1-b(8:11)*b(8:11)')^2
    
    % compute the gradients
    dfdV = zeros(d-1,d);
    dfdZ = -n*B.dF/B.F;
    for i=1:n
        for j=1:d-1
            dfdV(j,:) = dfdV(j,:) + 2*B.Z(j)*sum(SX(:,:,i) .* repmat(E(:,i).*(SX(:,:,i)*B.V(:,j)), [1,d])) / sum(E(:,i));
            dfdZ(j) = dfdZ(j) + E(:,i)'*(SX(:,:,i)*B.V(:,j)).^2 / sum(E(:,i));
        end            
    end
    dVdQ = compute_dVdQ(b(4:end));
    dfdQ = zeros(1,8);
    for i=1:3
        dfdQ = dfdQ + dfdV(i,:)*dVdQ(:,:,i);
    end
    dfdb = -[dfdZ, dfdQ] % + [4*(b(4:7)*b(4:7)'-1)*b(4:7), 4*(b(8:11)*b(8:11)'-1)*b(8:11)]]
end


function [c,ceq] = sb_constraints(b)

    c = 0;
    ceq = [0,0];
    
    if nargin == 1
        ceq = [1 - norm(b(3:7)), 1 - norm(b(8:11))];
    end
end

