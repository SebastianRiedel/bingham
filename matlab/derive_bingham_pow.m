%cd ~/bingham
%S = read_matrix('S3.txt');
%S = S(S(:,1)>=0,:);
n = size(S,1);
Q = S(:,1:4); V = S(:,5);

C1 = []; C2 = []; %C3 = [];
for i=1:10
    Q = Q*rand_orthogonal_matrix(4);
    W = Q(:,1); X = Q(:,2); Y = Q(:,3); Z = Q(:,4);
    U = Q(:,2:4) ./ repmat(sqrt(sum(Q(:,2:4).^2, 2)), [1 3]);
    U(U==NaN) = 0;
    W2 = W.^2; W3 = W.^3; W4 = W.^4; W5 = W.^5; W6 = W.^6; W7 = W.^7; W8 = W.^8;
    W03 = [ones(n,1), W, W2, W3];
    W04 = [ones(n,1), W, W2, W3, W4];
    W05 = [ones(n,1), W, W2, W3, W4, W5];
    W06 = [ones(n,1), W, W2, W3, W4, W5, W6];
    W07 = [ones(n,1), W, W2, W3, W4, W5, W6, W7];
    W08 = [ones(n,1), W, W2, W3, W4, W5, W6, W7, W8];
    A = 2*acos(W);
    A(A>pi) = A(A>pi) - 2*pi;
    S_weights = exp(-3*(1-W.^2));
    basis1 = W06;
    basis2 = W05;
    %basis1 = [W W3 W5].*repmat(X, [1 3]);
    %basis2 = [ones(n,1) W2 W4].*repmat(X.^2, [1 3]);
    %basis1 = [W W3].*repmat(X, [1 2]);
    %basis2 = [ones(n,1) W2].*repmat(X.^2, [1 2]);

    b_vals = .1:.1:2; %[.01:.01:.1, .15:.1:2, 2.25:.25:4]
    for j=1:length(b_vals)
        b = b_vals(j);
        fprintf('.');

        Qb = [cos(b*A/2), repmat(sin(b*A/2),[1,3]).*U];
        Wb = Qb(:,1); Xb = Qb(:,2); Yb = Qb(:,3); Zb = Qb(:,4);

        f = cos(b*A/2).*sin(b*A/2)./sin(A/2);
        %f = Wb.*Xb;
        C = weighted_ridge_regression(basis1, f, S_weights);
        R = basis1*C - f;
        C1(j,:,i) = C';
        idx = randperm(n,10000);
        figure(11); subplot(2,1,1); plot(W(idx),f(idx),'.'); hold on, plot(W(idx), basis1(idx,:)*C, 'r.'); hold off
        subplot(2,1,2); plot(W(idx).*S_weights(idx), R(idx), '.'); ylabel('weighted error'); ylim([-.2 .2])

        f = sin(b*A/2).^2 ./ sin(A/2).^2;
        %f = Xb.*Xb;
        C = weighted_ridge_regression(basis2, f, S_weights);
        R = basis2*C - f;
        C2(j,:,i) = C';
        figure(12); subplot(2,1,1); plot(W(idx),f(idx),'.'); hold on, plot(W(idx), basis2(idx,:)*C, 'r.'); hold off
        subplot(2,1,2); plot(W(idx).*S_weights(idx), R(idx), '.'); ylabel('weighted error'); ylim([-.2 .2])

        input(sprintf('b = %f:', b));
    end
    fprintf('\n');
end

figure(1); clf, hold on, for i=1:size(C1,3), plot(C1(:,:,i)); end, hold off
figure(2); clf, hold on, for i=1:size(C2,3), plot(C2(:,:,i)); end, hold off

C1 = mean(C1,3);
C2 = mean(C2,3);

for j=1:length(b_vals)-1
    b = (b_vals(j) + b_vals(j+1)) / 2;
    b0 = b_vals(j);
    Qb = [cos(b*A/2), repmat(sin(b*A/2),[1,3]).*U];
    Wb = Qb(:,1); Xb = Qb(:,2); Yb = Qb(:,3); Zb = Qb(:,4);
    Qb0 = [cos(b0*A/2), repmat(sin(b0*A/2),[1,3]).*U];
    Wb0 = Qb0(:,1); Xb0 = Qb0(:,2); Yb0 = Qb0(:,3); Zb0 = Qb0(:,4);

    f = Wb.*Xb;
    f0 = Wb0.*Xb0;
    C = (C1(j,:) + C1(j+1,:))' / 2;
    C0 = C1(j,:)';
    R = basis1*C - f;
    R0 = basis1*C0 - f0;
    idx = randperm(n,10000);
    figure(11); subplot(2,1,1); plot(W(idx).*S_weights(idx), R0(idx), '.'); ylabel('weighted error'); ylim([-.2 .2])
    subplot(2,1,2); plot(W(idx).*S_weights(idx), R(idx), '.'); ylabel('weighted error'); ylim([-.2 .2])

    f = Xb.*Xb;
    f0 = Xb0.*Xb0;
    C = (C2(j,:) + C2(j+1,:))' / 2;
    C0 = C2(j,:)';
    R = basis2*C - f;
    R0 = basis2*C0 - f0;
    figure(12); subplot(2,1,1); plot(W(idx).*S_weights(idx), R0(idx), '.'); ylabel('weighted error'); ylim([-.2 .2])
    subplot(2,1,2); plot(W(idx).*S_weights(idx), R(idx), '.'); ylabel('weighted error'); ylim([-.2 .2])
    
    input(sprintf('b = %f:', b));
end


% syms w b real
% f1 = cos(b*acos(w))*sin(b*acos(w))/sin(acos(w));
% f2 = sin(b*acos(w))^2/sin(acos(w))^2;
% 
% % 4th order taylor-series expansion of f1 about w=0
% c0 = subs(f1, 'w', 0);
% c1 = subs(diff(f1,w,1), 'w', 0);
% c2 = subs(diff(f1,w,2), 'w', 0);
% c3 = subs(diff(f1,w,3), 'w', 0);
% c4 = subs(diff(f1,w,4), 'w', 0);
% g1 = c0 + c1*w + c2*w^2/2 + c3*w^3/6 + c4*w^4/24
% 
% % 3rd order taylor-series expansion of f2 about w=0
% c0 = subs(f2, 'w', 0);
% c1 = subs(diff(f2,w,1), 'w', 0);
% c2 = subs(diff(f2,w,2), 'w', 0);
% c3 = subs(diff(f2,w,3), 'w', 0);
% g2 = c0 + c1*w + c2*w^2/2 + c3*w^3/6


% g1 = @(w,b) (w^2*(cos((pi*b)/2)*sin((pi*b)/2) - 4*b^2*cos((pi*b)/2)*sin((pi*b)/2)))/2 + (w^4*(9*cos((pi*b)/2)*sin((pi*b)/2) - 40*b^2*cos((pi*b)/2)*sin((pi*b)/2) + 16*b^4*cos((pi*b)/2)*sin((pi*b)/2)))/24 + cos((pi*b)/2)*sin((pi*b)/2) - w*(b*cos((pi*b)/2)^2 - b*sin((pi*b)/2)^2) - (w^3*(4*b^3*sin((pi*b)/2)^2 + 4*b*cos((pi*b)/2)^2 - 4*b*sin((pi*b)/2)^2 - 4*b^3*cos((pi*b)/2)^2))/6
% g2 = @(w,b) (w^3*(8*b^3*cos((pi*b)/2)*sin((pi*b)/2) - 14*b*cos((pi*b)/2)*sin((pi*b)/2)))/6 + (w^2*(2*sin((pi*b)/2)^2 - 2*b^2*sin((pi*b)/2)^2 + 2*b^2*cos((pi*b)/2)^2))/2 + sin((pi*b)/2)^2 - 2*b*w*cos((pi*b)/2)*sin((pi*b)/2)
% 
% figure(2);
% for b=.1:.1:3
%     w = -sqrt(.75):.01:.99;
%     f1 = cos(b*acos(w)).*sin(b*acos(w))./sin(acos(w));
%     f2 = sin(b*acos(w)).^2./sin(acos(w)).^2;
%     f1_taylor = [];
%     f2_taylor = [];
%     for i=1:length(w)
%         f1_taylor(i) = g1(w(i),b);
%         f2_taylor(i) = g2(w(i),b);
%     end        
%     plot(w, f1, 'b.');
%     hold on
%     plot(w, f2, 'k.');
%     plot(w, f1_taylor, 'r-');
%     plot(w, f2_taylor, 'r-');
%     hold off
%     title(sprintf('b = %.2f', b));
%     input(':');
% end



%figure(2); for b=[.01,.1,.9,1.5,2,2.5,3,3.5,4], w0=[-sqrt(.75),-.8,-.75,-.7], for i=1:4, subplot(1,4,i); w = w0(i):.01:.99; f1 = cos(b*acos(w)).*sin(b*acos(w))./sin(acos(w)); f2 = sin(b*acos(w)).^2./sin(acos(w)).^2; fobj1 = fit(w',f1','poly6'); fobj2 = fit(w',f2','poly5'); plot(fobj1, w, f1); hold on, plot(fobj2, w, f2, 'k.'); plot(w, f1.*sqrt(1-w.^2), 'b--'); plot(w, f2.*(1-w.^2), 'k--'); hold off, legend off, title(sprintf('b = %.2f, w0 = %.2f', b, w(1))); end, input(':'); end
%figure(2); for b=.1:.1:3, w0=[-sqrt(.75),-.8,-.75,-.7], for i=1:4, subplot(1,4,i); w = w0(i):.01:.99; f1 = cos(b*acos(w)).*sin(b*acos(w))./sin(acos(w)); f2 = sin(b*acos(w)).^2./sin(acos(w)).^2; f14 = fit(w',f1','poly4'); f23 = fit(w',f2','poly3'); f16 = fit(w',f1','poly6'); f25 = fit(w',f2','poly5'); f18 = fit(w',f1','poly8'); f27 = fit(w',f2','poly7'); plot(f14, w, f1); hold on, plot(f23, w, f2, 'k.'); plot(f16, w, f1, 'k.'); plot(f25, w, f2, 'k.'); plot(f18, w, f1, 'k.'); plot(f27, w, f2, 'k.'); plot(w, f1.*sqrt(1-w.^2), 'b--'); plot(w, f2.*(1-w.^2), 'k--'); hold off, legend off, title(sprintf('b = %.2f, w0 = %.2f', b, w(1))); end, input(':'); end


% syms s1 s2 s3 c1 c2 c3 real
% q = [c1, s1*c2, s1*s2*c3, s1*s2*s3];
% unique(kron(kron(q,q),kron(q,q)))'



