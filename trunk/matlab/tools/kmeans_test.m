
% Test a few clustering techniques and report the SSD under the L2 and
% Chi-Squared norms.

n = size(F,1);

X1 = F;
X2 = F ./ repmat(mean(F), n, 1);
X3 = F ./ repmat(sqrt(mean(F)), n, 1);


k = 5;  % num. clusters

[M1 L1] = kmeans(X1,k);
[M2 L2] = kmeans(X2,k);  %M2 = M2 .* repmat(mean(F), k, 1);
[M3 L3] = kmeans(X3,k);  %M3 = M3 .* repmat(sqrt(mean(F)), k, 1);

[M1_chi2 L1_chi2] = kmeans(X1,k);
[M2_chi2 L2_chi2] = kmeans(X2,k);  M2 = M2 .* repmat(mean(F), k, 1);
[M3_chi2 L3_chi2] = kmeans(X3,k);  M3 = M3 .* repmat(sqrt(mean(F)), k, 1);


% L2 SSD 1
ssd = 0;
X = X1; M = M1; L = L1;
for i=1:k
   Xi = X(find(L==i),:);
   Mi = repmat(M(i,:), [size(Xi,1) 1]);
   ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)));
end
ssd1 = ssd;

% L2 SSD 2
ssd = 0;
X = X1; M = M2; L = L2;
for i=1:k
   Xi = X(find(L==i),:);
   Mi = repmat(M(i,:), [size(Xi,1) 1]);
   ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)));
end
ssd2 = ssd;

% L2 SSD 3
ssd = 0;
X = X1; M = M3; L = L3;
for i=1:k
   Xi = X(find(L==i),:);
   Mi = repmat(M(i,:), [size(Xi,1) 1]);
   ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)));
end
ssd3 = ssd;


% Chi-Squared SSD 1
ssd = 0;
X = X1; M = M1_chi2; L = L1_chi2;
for i=1:k
   Xi = X(find(L==i),:);
   Mi = repmat(M(i,:), [size(Xi,1) 1]);
   ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)./(Xi + Mi)));
end
ssd1_chi2 = ssd;

% Chi-Squared SSD 2
ssd = 0;
X = X1; M = M2_chi2; L = L2_chi2;
for i=1:k
   Xi = X(find(L==i),:);
   Mi = repmat(M(i,:), [size(Xi,1) 1]);
   ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)./(Xi + Mi)));
end
ssd2_chi2 = ssd;

% Chi-Squared SSD 3
ssd = 0;
X = X1; M = M3_chi2; L = L3_chi2;
for i=1:k
   Xi = X(find(L==i),:);
   Mi = repmat(M(i,:), [size(Xi,1) 1]);
   ssd = ssd + sum(sum((Xi-Mi).*(Xi-Mi)./(Xi + Mi)));
end
ssd3_chi2 = ssd;


fprintf('\n\n');
fprintf('ssd1 = %f, ssd2 = %f, ssd3 = %f\n', ssd1, ssd2, ssd3);
fprintf('ssd1_chi2 = %f, ssd2_chi2 = %f, ssd3_chi2 = %f\n', ssd1_chi2, ssd2_chi2, ssd3_chi2);
fprintf('\n\n');


figure(10); k = size(M1,1); for i=1:k, subplot(k,1,i); bar(M1(i,:)); end
figure(11); k = size(M2,1); for i=1:k, subplot(k,1,i); bar(M2(i,:)); end
figure(12); k = size(M3,1); for i=1:k, subplot(k,1,i); bar(M3(i,:)); end

figure(13); k = size(M1,1); for c=1:k, subplot(k,1,c); for i=1:size(X1,1), if L1(i)==c, bar(X1(i,:)); pause(.01); end, end, end
figure(14); k = size(M2,1); for c=1:k, subplot(k,1,c); for i=1:size(X2,1), if L2(i)==c, bar(X2(i,:)); pause(.01); end, end, end
figure(15); k = size(M3,1); for c=1:k, subplot(k,1,c); for i=1:size(X3,1), if L3(i)==c, bar(X3(i,:)); pause(.01); end, end, end

