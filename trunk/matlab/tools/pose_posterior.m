function pmf = pose_posterior(Q, B, W, F, q_obs, f_obs)
% pmf = pose_posterior(Q, B, F, q_obs, f_obs) -- computes a posterior
% distribution over quaternions, Q, given bingham mixtures, B (with weights
% W), cluster feature centers, F, and observed quaternion and feature,
% q_obs and f_obs.

lambda = .05;

k = length(B);

for i=1:size(Q,1)   % cells
    
    % rotate the bingham mixture distributions by Q(i,:)
    %for c=1:k
    %    for j=1:length(B{c})
    %        B_rot{c}(j) = bingham_rotate_3d(B{c}(j), Q(i,:));
    %    end
    %end
    B_rot = B;

    % rotate q_obs by Q(i,:)
    %Qi_inv = [Q(i,1), -Q(i,2:4)];
    %q_obs_rot = quaternion_mult(q_obs, Qi_inv);
    %q_obs_rot = quaternion_mult(Qi_inv, q_obs);
    %q_obs_rot = quaternion_mult(q_obs_rot, Q(i,:));
    %q_obs_rot = q_obs;
    R1 = quaternionToRotationMatrix(Q(i,:));
    R2 = quaternionToRotationMatrix(q_obs);
    q_obs_rot = rotationMatrixToQuaternion(inv(R1)*R2);
    %input(':');
    
    mass = 0;
    for c=1:k
        p_pfh = lambda * exp(-lambda * norm(F(c,:) - f_obs));
        p_bingham = 0;
        for j=1:length(B_rot{c})
            p_bingham = p_bingham + W{c}(j)*bingham_pdf(q_obs_rot, B_rot{c}(j));
        end
        mass = mass + p_pfh * p_bingham;
        
        %fprintf('i=%d, c=%d: p_pfh = %f, p_bingham = %f\n', i, c, p_pfh, p_bingham);
    end

    %fprintf('mass = %f\n', mass);
    %input(':');
    
    pmf.mass(i) = mass;
end

pmf.mass = pmf.mass / sum(pmf.mass);
pmf.points = Q;
