%function F_cache = bingham_compute_F_cache()

Y = [ 0.00, 0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80,  0.90,  1.00, ...
 	  1.10,  1.20,  1.30,  1.40,  1.50,  1.60,  1.70,  1.80,  1.90,  2.00, ...
      2.20,  2.40,  2.60,  2.80,  3.00,  3.20,  3.40,  3.60,  3.80,  4.00, ...
      4.50,  5.00,  5.50,  6.00,  6.50,  7.00,  7.50,  8.00,  8.50,  9.00, ...
      9.50, 10.00, ...
    10.50, 11.00, 11.50, 12.00, 12.50, 13.00, 13.50, 14.00, ...
    14.50, 15.00, 15.50, 16.00, 16.50, 17.00, 17.50, 18.00, 18.50, 19.00, ...
    19.50, 20.00, 21.00, 22.00, 23.00, 24.00, 25.00, 26.00, 27.00, 28.00, ...
    29.00, 30.00];

% (negative) concentration parameters table
Z = Y.^2;
NZ = length(Z);
F_cache.Z = Z;





%========================================================================%

%-------------  log(1F1), log(2F1), log(3F1) lookup tables  -------------%

%========================================================================%


% log 1F1(a; b; z) for z >= 0
A = [.5, 1.5, 2.5]; B = .5:.5:3*max(Z);
F_cache.log1F1 = zeros(length(A), length(B), length(Z));
for ai=1:length(A), a = A(ai);
    for bi=1:length(B), b = B(bi); n = 10; fprintf('.');
        for zi=1:length(Z), z = Z(zi);
            while 1
                i = 0:n-1; ilogz = i*log(z); if z==0, ilogz(1) = 0; end
                %=====================================================%
                X = ilogz - gammaln(i+1) + gammaln(i+a) - gammaln(i+b);
                %=====================================================%
                xmax = max(X); if X(end) < xmax-16, break, end, n = n+10;
            end
            F_cache.log1F1(ai,bi,zi) = xmax + log(sum(exp(X-xmax))) + log(sqrt(pi));
        end
    end
    fprintf('\n');
end
save F_cache.mat F_cache


% log 2F1(a1,a2; b; z1,z1-z2) for z1 >= z2 >= 0
A = [.5, 1.5, 2.5]; B = .5:.5:2*max(Z); nmax = 0;
F_cache.log2F1 = zeros(length(A), length(A), length(B), length(Z), length(Z));
for ai1=1:length(A), a1 = A(ai1);
    for ai2=1:length(A), a2 = A(ai2);      %if ai1>1 || ai2==3, continue, end  %dbug!
        for bi=1:length(B), b = B(bi); fprintf('.'); n = 10;
            for zi1=1:length(Z)
                for zi2=1:zi1, dz = Z(zi1) - Z(zi2);
                    while 1
                        i = 0:n-1; ilogdz = i*log(dz); if dz==0, ilogdz(1) = 0; end
                        %============================================================================%
                        X = ilogdz - gammaln(i+1) + gammaln(i+a2) + F_cache.log1F1(ai1, 2*(b+i), zi1);
                        %============================================================================%
                        xmax = max(X); if X(end) < xmax-16, break, end, n = n+10; if n > nmax, nmax = n; end
                    end
                    F_cache.log2F1(ai1,ai2,bi,zi1,zi2) = xmax + log(sum(exp(X-xmax)));
                end
            end
            fprintf('bi=%d, n=%d\n', bi, n);
        end
        save F_cache.mat F_cache
        fprintf('\n');
    end
end


% log 3F1(a1,a2,a3; b; z1,z1-z2,z1-z3) for z1 >= z2 >= z3 >= 0 and b = a1+a2+a3+.5
A = [.5, 1.5, 2.5]; NA = length(A); nmax = 0;
F_cache.log3F1 = zeros(NA,NA,NA,NZ,NZ,NZ);
for ai1=1:NA, a1 = A(ai1);
    for ai2=1:NA-ai1+1, a2 = A(ai2);
        for ai3=1:NA-ai1-ai2+2, a3 = A(ai3);
            b = a1+a2+a3+.5; n = 10;
            for zi1=1:length(Z), fprintf('.');
                for zi2=1:zi1, dz = Z(zi1) - Z(zi2);
                    for zi3=1:zi2
                        while 1
                            i = 0:n-1; ilogdz = i*log(dz); if dz==0, ilogdz(1) = 0; end
                            %====================================================================================================%
                            X = ilogdz - gammaln(i+1) + gammaln(a2+i) + reshape(F_cache.log2F1(ai1,ai3, 2*(b+i), zi1,zi3), [1,n]);
                            %====================================================================================================%
                            xmax = max(X); if X(end) < xmax-16, break, end, n = n+10; if n > nmax, nmax = n; end
                        end
                        F_cache.log3F1(ai1,ai2,ai3,zi1,zi2,zi3) = xmax + log(sum(exp(X-xmax)));
                    end
                end
                fprintf('zi1=%d, n=%d\n', zi1, n);
            end
        end
    end
end




%==================================================================%

%------------------  Normalization Constant (F)  ------------------%

%==================================================================%

% 2*1F1(1/2; 1; z) for z <= 0
F_cache.F{1} = 2*exp(reshape(F_cache.log1F1(1,2,:),[1,NZ]) - Z);   % 2*exp(z)*1F1(a; b; -z)

% 2*1F1(1/2; 3/2; z1,z2) for z1 <= z2 <= 0
F_cache.F{2} = 2*exp(reshape(F_cache.log2F1(1,1,3,:,:),[NZ,NZ]) - repmat(Z',[1,NZ]));  % 2*exp(z1)*2F1(a,a; b; -z1, z2-z1)
for z=1:NZ-1
    F_cache.F{2}(z,z+1:end) = 0;
end

% 2*1F1(1/2; 4/2; z1,z2,z3) for z1 <= z2 <= z3 <= 0
F_cache.F{3} = 2*exp(reshape(F_cache.log3F1(1,1,1,:,:,:),[NZ,NZ,NZ]) - repmat(Z', [1,NZ,NZ]));  % 2*exp(z1)*2F1(a,a,a; b; -z1, z2-z1, z3-z1)
for z1=1:NZ-1
    F_cache.F{3}(z1,z1+1:end,:) = 0;
    for z2=1:z1
        F_cache.F{3}(z1,z2,z2+1:end) = 0;
    end
end

save F_cache.mat F_cache




%=============================================================%

%-----------------  First-order Derivatives  -----------------%

%=============================================================%

% (d/dz) 2*1F1(1/2; 1; z) for z <= 0
b = 2/2;
F = F_cache.F{1};
dF1 = 2*exp(reshape(F_cache.log1F1(2,2*(b+1),:),[1,NZ]) - Z);  % 2*exp(z)*1F1(a+1; b+1; -z)
F_cache.dF{1} = F - dF1;

% (d/dZ) 2*1F1(1/2; 3/2; z1,z2) for z1 <= z2 <= 0
b = 3/2;
F = reshape(F_cache.F{2}, [1,NZ,NZ]);
dF1 = 2*exp(reshape(F_cache.log2F1(2,1,2*(b+1),:,:),[1,NZ,NZ]) - repmat(Z,[1,1,NZ]));  % 2*exp(z1)*2F1(a+1,a; b+1; -z1, z2-z1)
dF2 = 2*exp(reshape(F_cache.log2F1(1,2,2*(b+1),:,:),[1,NZ,NZ]) - repmat(Z,[1,1,NZ]));  % 2*exp(z1)*2F1(a,a+1; b+1; -z1, z2-z1)
F_cache.dF{2}(1,:,:) = F - dF1 - dF2;
F_cache.dF{2}(2,:,:) = dF2;
for z=1:NZ-1
    F_cache.dF{2}(:,z,z+1:end) = 0;
end

% (d/dZ) 2*1F1(1/2; 4/2; z1,z2,z3) for z1 <= z2 <= z3 <= 0
F = reshape(F_cache.F{3}, [1,NZ,NZ,NZ]);
dF1 = 2*exp(reshape(F_cache.log3F1(2,1,1,:,:,:),[1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a+1,a,a; b+1; -z1, z2-z1, z3-z1)
dF2 = 2*exp(reshape(F_cache.log3F1(1,2,1,:,:,:),[1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a,a+1,a; b+1; -z1, z2-z1, z3-z1)
dF3 = 2*exp(reshape(F_cache.log3F1(1,1,2,:,:,:),[1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a,a,a+1; b+1; -z1, z2-z1, z3-z1)
F_cache.dF{3}(1,:,:,:) = F - dF1 - dF2 - dF3;
F_cache.dF{3}(2,:,:,:) = dF2;
F_cache.dF{3}(3,:,:,:) = dF3;
for z1=1:NZ-1
    F_cache.dF{3}(:,z1,z1+1:end,:) = 0;
    for z2=1:z1
        F_cache.dF{3}(:,z1,z2,z2+1:end) = 0;
    end
end

save F_cache.mat F_cache




%==============================================================%

%-----------------  Second-order Derivatives  -----------------%

%==============================================================%


% (d^2/dz^2) 1F1(1/2; 2/2; z) for z <= 0
b = 2/2;
F = F_cache.F{1};
dF1 = F_cache.dF{1};
ddF11 = 2*exp(reshape(F_cache.log1F1(3,2*(b+2),:), [1,NZ]) - Z);  % 2*exp(z)*1F1(a+2; b+2; -z)
F_cache.ddF{1} = -F + 2*dF1 + ddF11;

% (d/dZ) 2*1F1(1/2; 3/2; z1,z2) for z1 <= z2 <= 0
b = 3/2;
F = reshape(F_cache.F{2}, [1,NZ,NZ]);
dF1 = F_cache.dF{2}(1,:,:);
dF2 = F_cache.dF{2}(2,:,:);
ddF11 = 2*exp(reshape(F_cache.log2F1(3,1,2*(b+2),:,:), [1,NZ,NZ]) - repmat(Z,[1,1,NZ]));  % 2*exp(z1)*2F1(a+2,a; b+2; -z1, z2-z1)
ddF12 = 2*exp(reshape(F_cache.log2F1(2,2,2*(b+2),:,:), [1,NZ,NZ]) - repmat(Z,[1,1,NZ]));  % 2*exp(z1)*2F1(a+1,a+1; b+2; -z1, z2-z1)
ddF22 = 2*exp(reshape(F_cache.log2F1(1,3,2*(b+2),:,:), [1,NZ,NZ]) - repmat(Z,[1,1,NZ]));  % 2*exp(z1)*2F1(a,a+1; b+2; -z1, z2-z1)
F_cache.ddF{2}(1,:,:) = -F + 2*dF1 + ddF11 + 2*ddF12 + ddF22;
F_cache.ddF{2}(2,:,:) = dF2 - ddF12 - ddF22;
F_cache.ddF{2}(3,:,:) = ddF22;
for z=1:NZ-1
    F_cache.ddF{2}(:,z,z+1:end) = 0;
end

% (d^2/dZ^2) 1F1(1/2; 4/2; z1,z2,z3) for z1 <= z2 <= z3 <= 0
b = 4/2;
F = reshape(F_cache.F{3}, [1,NZ,NZ,NZ]);
dF1 = F_cache.dF{3}(1,:,:,:);
dF2 = F_cache.dF{3}(2,:,:,:);
dF3 = F_cache.dF{3}(3,:,:,:);
ddF11 = 2*exp(reshape(F_cache.log3F1(3,1,1,:,:,:), [1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a+2,a,a; b+2; -z1, z2-z1, z3-z1)
ddF12 = 2*exp(reshape(F_cache.log3F1(2,2,1,:,:,:), [1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a+1,a+1,a; b+2; -z1, z2-z1, z3-z1)
ddF13 = 2*exp(reshape(F_cache.log3F1(2,1,2,:,:,:), [1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a+1,a,a+1; b+2; -z1, z2-z1, z3-z1)
ddF22 = 2*exp(reshape(F_cache.log3F1(1,3,1,:,:,:), [1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a,a+2,a; b+2; -z1, z2-z1, z3-z1)
ddF23 = 2*exp(reshape(F_cache.log3F1(1,2,2,:,:,:), [1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a,a+1,a+1; b+2; -z1, z2-z1, z3-z1)
ddF33 = 2*exp(reshape(F_cache.log3F1(1,1,3,:,:,:), [1,NZ,NZ,NZ]) - repmat(Z,[1,1,NZ,NZ]));  % 2*exp(z1)*3F1(a,a,a+2; b+2; -z1, z2-z1, z3-z1)
F_cache.ddF{3}(1,:,:,:) = -F + 2*dF1 + ddF11 + 2*ddF12 + 2*ddF13 + ddF22 + 2*ddF23 + ddF33;
F_cache.ddF{3}(2,:,:,:) = dF2 - ddF12 - ddF22 - ddF23;
F_cache.ddF{3}(3,:,:,:) = dF3 - ddF13 - ddF23 - ddF33;
F_cache.ddF{3}(4,:,:,:) = ddF22;
F_cache.ddF{3}(5,:,:,:) = ddF23;
F_cache.ddF{3}(6,:,:,:) = ddF33;
for z1=1:NZ-1
    F_cache.ddF{3}(:,z1,z1+1:end,:) = 0;
    for z2=1:z1
        F_cache.ddF{3}(:,z1,z2,z2+1:end) = 0;
    end
end

save F_cache.mat F_cache




%=============================================================%

%-----------------  Third-order Derivatives  -----------------%

%=============================================================%


% % (d^2/dz^2) 1F1(1/2; 2/2; z) for z <= 0
% b = 2/2;
% F = F_cache.F{1};
% dF1 = F_cache.dF{1};
% ddF11 = F_cache.ddF{1};
% F_cache.dddF111{1} = -F + 2*dF1 + ddF11;





bingham_constants = [];
bingham_constants.Z = Z;
bingham_constants.F = F_cache.F;
bingham_constants.dF = F_cache.dF;
bingham_constants.ddF = F_cache.ddF;
save bingham_constants.mat bingham_constants








