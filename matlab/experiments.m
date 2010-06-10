
% get CV error for the IKEA-Synthetic-k10
objs = ikea_objects_synth();
for i=1:length(objs)
   [E_med_i E_mean_i C_mean_i] = get_tofoo_error(sprintf('../../my_papers/jglov/rss10/exp/IKEA-Synthetic-k10/%s', objs{i}), 50);
   E_med(:,:,:,i) = E_med_i;
   E_mean(:,:,:,i) = real(E_mean_i);
   C_mean(:,:,:,i) = C_mean_i;
end

figure(7); plot(mean(E_med,4), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');
figure(8); plot(mean(E_mean,4), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');
figure(9); plot(mean(C_mean,4), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');


% get CV error for the IKEA-Synthetic

%cd C:\cygwin\home\jglov\my_papers\jglov\rss10\exp\IKEA-Synthetic-k10\
cd C:\Doc'uments and Settings'\jglov\My' Documents'\Downloads\gazebo
obj_files = dir('.');
objs = {}; for i=3:length(obj_files); objs{i-2} = obj_files(i).name; end
%objs = ikea_objects_synth();
clear E_med E_mean C_mean;
for i=1:length(objs)
   fprintf('obj %d -- %s', i, objs{i});
   [E_med_i E_mean_i C_mean_i] = get_tofoo_error(sprintf('%s', objs{i}), 50, 1);
   E_med(:,:,:,i) = E_med_i;
   E_mean(:,:,:,i) = real(E_mean_i);  %dbug
   C_mean(:,:,:,i) = C_mean_i;
end

E_med = reshape(E_med, [size(E_med,1) size(E_med,3) size(E_med,4)]);
E_mean = reshape(E_mean, [size(E_mean,1) size(E_mean,3) size(E_mean,4)]);
C_mean = reshape(C_mean, [size(C_mean,1) size(C_mean,3) size(C_mean,4)]);

%dbug
idx = [1 3 6 8 13 14 18 20 23 34 37 40 41 42];
E_med = E_med(:,:,idx);
E_mean = E_mean(:,:,idx);
C_mean = C_mean(:,:,idx);

E_base = zeros(10000,20);
x = rand(10000,20)*pi/2; for i=1:size(x,1), for j=1:size(x,2), E_base(i,j) = (180/pi)*min(x(i,1:j)); end, end
E_med_base = median(E_base);
E_mean_base = mean(E_base);

figure(1); errorbar(mean(mean(E_med,3),2), std(mean(E_med,3),0,2), 'k-'); hold on; plot(E_med_base, 'r--'); hold off;
figure(2); errorbar(mean(mean(E_mean,3),2), std(mean(E_mean,3),0,2), 'k-'); hold on; plot(E_mean_base, 'r--'); hold off;
figure(3); plot(mean(C_mean,3), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');

for i=1:size(E_med,3)
   %pcd_model = load_pcd(sprintf('%s/%s/cv1.pcd', fdir, objs{i}));
   %figure(4);
   %plot_pcd(pcd_model);
   figure(5);
   plot(mean(E_med(:,:,i),2));
   title(sprintf('obj %d -- %s', i, objs{i}));
   figure(6);
   plot(mean(E_mean(:,:,i),2));
   title(sprintf('obj %d -- %s', i, objs{i}));
   input(':');
end


