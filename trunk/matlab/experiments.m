
% get CV error for the IKEA-Synthetic-k10
objs = ikea_objects_synth();
for i=1:length(objs)
   [E_med_i E_mean_i C_mean_i] = get_tofoo_error(sprintf('../../IKEA-Synthetic-k10/%s', objs{i}), 50);
   E_med(:,:,:,i) = E_med_i;
   E_mean(:,:,:,i) = real(E_mean_i);
   C_mean(:,:,:,i) = C_mean_i;
end

figure(7); plot(mean(E_med,4), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');
figure(8); plot(mean(E_mean,4), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');
figure(9); plot(mean(C_mean,4), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');


% get CV error for the IKEA-Synthetic
objs = ikea_objects_synth();
clear E_med E_mean C_mean;
for i=1:length(objs)
   [E_med_i E_mean_i C_mean_i] = get_tofoo_error(sprintf('../../IKEA-Synthetic/%s', objs{i}), 50);
   E_med(:,:,:,i) = E_med_i;
   E_mean(:,:,:,i) = real(E_mean_i);  %dbug
   C_mean(:,:,:,i) = C_mean_i;
end

E_med = reshape(E_med, [size(E_med,1) size(E_med,3) size(E_med,4)]);
E_mean = reshape(E_mean, [size(E_mean,1) size(E_mean,3) size(E_mean,4)]);
C_mean = reshape(C_mean, [size(C_mean,1) size(C_mean,3) size(C_mean,4)]);

figure(1); plot(mean(E_med,3), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');
figure(2); plot(mean(E_mean,3), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');
figure(3); plot(mean(C_mean,3), 'LineWidth', 2); legend('k=1', 'k=2', 'k=3', 'k=4', 'k=5');

