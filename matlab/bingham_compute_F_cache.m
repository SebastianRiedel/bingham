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
F_cache.Z = Z;

%if 0 %dbug

% 1F1(1/2; 1; z) for z<0
F_cache.table{1} = zeros(1, length(Z));
a = 1/2;
b = 1;
n = 10;
F_cache.table{1}(1) = 2*sqrt(pi)*gamma(a)/gamma(b);  % z=0
for zi=2:length(Z)
    z = Z(zi);
    logz = log(z);
    while 1
        i = 0:n-1;
        X = gammaln(i+a) - gammaln(i+b) + i*logz - gammaln(i+1);
        c = max(X);
        if X(end) < c-16
            break
        end
        n = n+10;
    end
    %F = 2*sqrt(pi)*exp(c)*sum(exp(X-c))
    logF = log(2*sqrt(pi)) + c + log(sum(exp(X-c)));
    F_cache.table{1}(zi) = exp(logF - z);
end


% log 1F1(1/2; b; z) for z>0
F_cache.btable{1} = zeros(6*max(Z), length(Z));
for b=.5:.5:3*max(Z)
    fprintf('.');
    a = 1/2;
    F_cache.btable{1}(2*b,1) = log(2*sqrt(pi)) + gammaln(a) - gammaln(b);  % z=0
    n = 10;
    for zi=2:length(Z)
        z = Z(zi);
        logz = log(z);
        while 1
            i = 0:n-1;
            X = gammaln(i+a) - gammaln(i+b) + i*logz - gammaln(i+1);
            xmax = max(X);
            if X(end) < xmax-16
                break
            end
            n = n+10;
        end
        %F = 2*sqrt(pi)*exp(c)*sum(exp(X-c))
        logF = xmax + log(sum(exp(X-xmax))) + log(2*sqrt(pi));
        F_cache.btable{1}(2*b,zi) = logF;
        %[b,z,logF]
    end
end
fprintf('\n');    


% log 1F1(1/2; b; z1-z2) for z1>z2>0
F_cache.btable{2} = zeros(6*max(Z), length(Z), length(Z));
for b=.5:.5:3*max(Z)   %.5:.5:2*max(Z)
    fprintf('.');
    a = 1/2;
    n = 10;
    for zi1=1:length(Z)
        z1 = Z(zi1);
        for zi2=1:zi1-1
            z2 = Z(zi2);
            z = z1-z2;
            logz = log(z);
            while 1
                i = 0:n-1;
                X = gammaln(i+a) - gammaln(i+b) + i*logz - gammaln(i+1);
                xmax = max(X);
                if X(end) < xmax-16
                    break
                end
                n = n+10;
            end
            %F = 2*sqrt(pi)*exp(c)*sum(exp(X-c))
            logF = xmax + log(sum(exp(X-xmax))) + log(2*sqrt(pi));
            F_cache.btable{2}(2*b,zi1,zi2) = logF;
            %[b,z,logF]
        end
        F_cache.btable{2}(2*b,zi1,zi1) = F_cache.btable{1}(2*b,1);  % z1=z2
    end
end
fprintf('\n');    


% 1F1(1/2; 3/2; z1,z2) for z1,z2 < 0
F_cache.table{2} = zeros(length(Z), length(Z));
a = 1/2;
b = 3/2;
nmax = 0;
for zi=1:length(Z)
    z = Z(zi);
    logz = log(z);
    F_cache.table{2}(zi,zi) = exp(gammaln(a) + F_cache.btable{1}(2*b,zi) - z);
    n = 10;
    for zi2=1:zi-1
        z2 = Z(zi2);
        while 1
            i = 0:n-1;
            bi = 2*(b+i);
            X = gammaln(i+a) + i*logz - gammaln(i+1) + F_cache.btable{2}(bi,zi,zi2)';
            xmax = max(X);
            if X(end) < xmax-16
                break
            end
            n = n + 10;
            if n > nmax
                nmax = n
            end
        end
        %F = exp(xmax)*sum(exp(X-xmax))
        logF = xmax + log(sum(exp(X-xmax)));
        F_cache.table{2}(zi,zi2) = exp(logF - z);
        %[z,z2,logF-z]
        %plot(X);
        %input(':');
    end
end
%F_cache.table{2}



% 1F1(1/2; 4/2; z1,z2,z3) for z1,z2,z3 < 0
F_cache.table{3} = zeros(length(Z), length(Z), length(Z));
a = 1/2;
b = 2;  % (d+1)/2
nmax = 0;
for zi1=1:length(Z)
    fprintf('.');
    z1 = Z(zi1);
    logz1 = log(z1);
    F_cache.table{3}(zi1,zi1,zi1) = exp(2*gammaln(a) + F_cache.btable{1}(2*b,zi1) - z1);  % z1=z2=z3
    n1 = 10;
    for zi2=1:zi1
        %z2 = Z(zi2);
        %logz2 = log(z1-z2);
        n2 = 10;
        for zi3=1:zi2
            if zi1==zi3  % z1=z2=z3
                continue;
            end
            z3 = Z(zi3);
            logz3 = log(z1-z3);
            while 1
                [I2,I1] = meshgrid(0:n2-1, 0:n1-1);
                BI = 2*(b+I1+I2);
                X = gammaln(I1+a) + gammaln(I2+a) + I1*logz1 + I2*logz3 - gammaln(I1+1) - gammaln(I2+1);
                X = X + reshape(F_cache.btable{2}(BI(1:end),zi1,zi2), [n1,n2]);
                xmax = max(max(X));
                if max(X(end,:)) < xmax-16 && max(X(:,end)) < xmax-16
                    break
                end
                if max(X(end,:)) > max(X(:,end))
                    n1 = n1+10;
                    if n1 > nmax
                        nmax = n1
                    end
                else
                    n2 = n2+10;
                    if n2 > nmax
                        nmax = n2
                    end
                end
            end
            %F = exp(c)*sum(exp(X-c))
            logF = xmax + log(sum(sum(exp(X-xmax))));
            F_cache.table{3}(zi1,zi2,zi3) = exp(logF - z1);
            %[b,z,z2,logF]
            %plot(X);  %hold on;  plot(F_cache.table{1}(bi2,zi2), 'r-');  hold off;
            %input(':');
        end
    end
end

%end %dbug


% (d/dZ) 1F1(1/2; 2/2; z) for z < 0
F_cache.dF{1} = zeros(1,length(Z));
a = 1/2;
b = 2/2;
n = 10;
F_cache.dF{1}(1) = 2*sqrt(pi)*gamma(a+1)/gamma(b+1);
for zi=2:length(Z)
    z = Z(zi);
    logz = log(z);
    % compute dF1
    while 1
        i = 0:n-1;
        X = gammaln(a+i+1) - gammaln(b+i+1) + log(i+1) + i*logz - gammaln(i+2);
        xmax = max(X);
        if X(end) < xmax-16
            break
        end
        n = n + 10;
    end
    %F = 2*sqrt(pi)*exp(xmax)*sum(exp(X-xmax))
    logF = log(2*sqrt(pi)) + xmax + log(sum(exp(X-xmax)));
    dF1 = exp(logF - z);

    F_cache.dF{1}(zi) = F_cache.table{1}(zi) - dF1;
end
%F_cache.dF{1}

%end %dbug

% (d/dZ) 1F1(1/2; 3/2; z1,z2) for z1,z2 < 0
F_cache.dF{2} = zeros(2, length(Z), length(Z));
a = 1/2;
b = 3/2;
F_cache.dF{2}(:,1,1) = 2*sqrt(pi)*gamma(a)*gamma(a+1)/gamma(b+1); % z1=z2=0
nmax = 0;
for zi=2:length(Z)
    z = Z(zi);
    logz = log(z);
    n = 10;
    for zi2=1:zi
        
        % compute dF1
        while 1
            i = 0:n-1;
            bi = 2*(b+i+1);
            X = gammaln(a+i+1) + log(i+1) + i*logz - gammaln(i+2) + F_cache.btable{2}(bi,zi,zi2)';
            xmax = max(X);
            if X(end) < xmax-16
                break
            end
            n = n + 10;
            if n > nmax
                nmax = n
            end
        end
        %F = exp(xmax)*sum(exp(X-xmax))
        logF = xmax + log(sum(exp(X-xmax)));
        dF1 = exp(logF - z);
        
        % compute dF2
        if zi==zi2
            logF = gammaln(a+1) + F_cache.btable{1}(2*(b+1), zi);
            dF2 = exp(logF - z);
        else
            z2 = Z(zi2);
            logdz = log(z-z2);
            while 1
                i = 0:n-1;
                bi = 2*(b+i+1);
                X = gammaln(a+i+1) + log(i+1) + i*logdz - gammaln(i+2) + F_cache.btable{1}(bi,zi)';
                xmax = max(X);
                if X(end) < xmax-16
                    break
                end
                n = n + 10;
                if n > nmax
                    nmax = n
                end
            end
            %F = exp(xmax)*sum(exp(X-xmax))
            logF = xmax + log(sum(exp(X-xmax)));
            dF2 = exp(logF - z);
        end
        
        F_cache.dF{2}(1,zi,zi2) = F_cache.table{2}(zi,zi2) - dF1 - dF2;
        F_cache.dF{2}(2,zi,zi2) = dF2;
        
        %[z,z2,logF-z]
        %plot(X);
        %input(':');
    end
end
%F_cache.dF{2}



% (d/dZ) 1F1(1/2; 4/2; z1,z2,z3) for z1,z2,z3 < 0
F_cache.dF{3} = zeros(3, length(Z), length(Z), length(Z));
a = 1/2;
b = 4/2;
F_cache.dF{3}(:,1,1,1) = 2*sqrt(pi)*gamma(a)^2*gamma(a+1)/gamma(b+1); % z1=z2=z3=0
nmax = 0;
for zi1=2:length(Z)
    z1 = Z(zi1);
    logz1 = log(z1);
    fprintf('zi1=%d, z1=%f\n', zi1, z1);
    for zi2=1:zi1
        fprintf('.');
        z2 = Z(zi2);
        logz2 = log(z1-z2);
        n1 = 10;
        n2 = 10;
        for zi3=1:zi2
            z3 = Z(zi3);
            logz3 = log(z1-z3);
        
            % compute dF1
            while 1
                [I2,I1] = meshgrid(0:n2-1, 0:n1-1);
                BI = 2*(b+I1+I2+1);
                if z1==z3
                    X = repmat(-inf,[n1,n2]);
                    X0 = gammaln(a+I1+1) + gammaln(a+I2) + log(I1+1) + I1*logz1 - gammaln(I1+2) - gammaln(I2+1);
                    X(~I2) = X0(~I2);
                else
                    X = gammaln(a+I1+1) + gammaln(a+I2) + log(I1+1) + I1*logz1 + I2*logz3 - gammaln(I1+2) - gammaln(I2+1);
                end
                X = X + reshape(F_cache.btable{2}(BI(1:end),zi1,zi2), [n1,n2]);
                xmax = max(max(X));
                if max(X(end,:)) < xmax-16 && max(X(:,end)) < xmax-16
                    break
                end
                if max(X(end,:)) > max(X(:,end))
                    n1 = n1+10;
                    if n1 > nmax
                        nmax = n1
                    end
                else
                    n2 = n2+10;
                    if n2 > nmax
                        nmax = n2
                    end
                end
            end
            %F = exp(c)*sum(exp(X-c))
            logF = xmax + log(sum(sum(exp(X-xmax))));
            dF1 = exp(logF - z1);

            % compute dF2
            while 1
                [I2,I1] = meshgrid(0:n2-1, 0:n1-1);
                BI = 2*(b+I1+I2+1);
                if z1==z2
                    X = repmat(-inf,[n1,n2]);
                    X0 = gammaln(a+I1+1) + gammaln(a+I2) + log(I1+1) + I2*logz1 - gammaln(I1+2) - gammaln(I2+1);
                    X(~I1) = X0(~I1);
                else
                    X = gammaln(a+I1+1) + gammaln(a+I2) + log(I1+1) + I1*logz2 + I2*logz1 - gammaln(I1+2) - gammaln(I2+1);
                end
                X = X + reshape(F_cache.btable{2}(BI(1:end),zi1,zi3), [n1,n2]);
                xmax = max(max(X));
                if max(X(end,:)) < xmax-16 && max(X(:,end)) < xmax-16
                    break
                end
                if max(X(end,:)) > max(X(:,end))
                    n1 = n1+10;
                    if n1 > nmax
                        nmax = n1
                    end
                else
                    n2 = n2+10;
                    if n2 > nmax
                        nmax = n2
                    end
                end
            end
            %F = exp(c)*sum(exp(X-c))
            logF = xmax + log(sum(sum(exp(X-xmax))));
            dF2 = exp(logF - z1);
        
            % compute dF3
            while 1
                [I2,I1] = meshgrid(0:n2-1, 0:n1-1);
                BI = 2*(b+I1+I2+1);
                if z1==z3
                    X = repmat(-inf,[n1,n2]);
                    X0 = gammaln(a+I1+1) + gammaln(a+I2) + log(I1+1) + I2*logz1 - gammaln(I1+2) - gammaln(I2+1);
                    X(~I1) = X0(~I1);
                else
                    X = gammaln(a+I1+1) + gammaln(a+I2) + log(I1+1) + I1*logz3 + I2*logz1 - gammaln(I1+2) - gammaln(I2+1);
                end
                X = X + reshape(F_cache.btable{2}(BI(1:end),zi1,zi2), [n1,n2]);
                xmax = max(max(X));
                if max(X(end,:)) < xmax-16 && max(X(:,end)) < xmax-16
                    break
                end
                if max(X(end,:)) > max(X(:,end))
                    n1 = n1+10;
                    if n1 > nmax
                        nmax = n1
                    end
                else
                    n2 = n2+10;
                    if n2 > nmax
                        nmax = n2
                    end
                end
            end
            %F = exp(c)*sum(exp(X-c))
            logF = xmax + log(sum(sum(exp(X-xmax))));
            dF3 = exp(logF - z1);
        
            F_cache.dF{3}(1,zi1,zi2,zi3) = F_cache.table{3}(zi1,zi2,zi3) - dF1 - dF2 - dF3;
            F_cache.dF{3}(2,zi1,zi2,zi3) = dF2;
            F_cache.dF{3}(3,zi1,zi2,zi3) = dF3;
        
            %[z,z2,logF-z]
            %plot(X);
            %input(':');
        end
    end
    if zi1==50 || zi1==60 || zi1==70
        save F_cache.mat F_cache
    end
end
%F_cache.dF{3}


%end %dbug


% fill in the diagonals (2-D)
%b = 3/2;
%for zi=1:length(Z)
%    F_cache.table{2}(zi,zi) = sqrt(pi)*F_cache.table{1}(zi);
%    
%    F_cache.dF{2}(1,zi,zi) = sqrt(pi)*F_cache.dF{1}(zi);
%    F_cache.table{2}(zi,zi) = .5*sqrt(pi)*F_cache.btable{1}(2*(b+1),zi);
%     = 
%    
%end

% fill in the diagonals (3-D)
%for zi=1:length(Z)
%    for zi3=1:zi
%        F_cache.table{3}(zi,zi,zi3) = sqrt(pi)*F_cache.table{2}(zi,zi3);
%        F_cache.dF{3}
%    end
%end

% F(-z1,-z1,-z3) -> F(z1,z1-z3,0) -> sqrt(pi)*F(z1,z1-z3) -> sqrt(pi)*F(-z1,-z3)


