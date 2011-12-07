function [X2,Q2,W2] = aggregate_poses(X,Q,W)
% [X2,Q2,W2] = aggregate_poses(X,Q,W)

rot_symm = 1;
pose_agg_x = 30;
pose_add_q = .3;

n = length(W);

cnt = 0;

for i=1:n
    R1 = quaternion_to_rotation_matrix(Q(i,:));
    z1 = R1(:,3);
    for j=1:cnt
        dx = norm(X(i,:) - X2(j,:));
        if rot_symm
            R2 = quaternion_to_rotation_matrix(Q2(j,:));
            z2 = R2(:,3);
        	dq = acos(dot(z1, z2));
        else
            dq = acos(fabs(dot(Q(i,:), Q2(j,:))));
        end
        if (dx < pose_agg_x && dq < pose_agg_q)   % add pose i to cluster j
            wtot = W(i) + W2(j);
            w = W(i) / wtot;
	wavg(agg_poses->X[j], poses->X[j], agg_poses->X[i], w, 3);
	if (olf->rot_symm) {
	  wavg(z, z1, z2, w, 3);
	  normalize(z, z, 3);
	  if (1 - z[2] < .00000001) {  // close to identity rotation
	    agg_poses->Q[j][0] = 1;
	    agg_poses->Q[j][1] = 0;
	    agg_poses->Q[j][2] = 0;
	    agg_poses->Q[j][3] = 0;
	  }
	  else {
	    double a = 1.0 / sqrt(1 - z[2]*z[2]);
	    double c = sqrt((1 + z[2])/2.0);
	    double s = sqrt((1 - z[2])/2.0);
	    agg_poses->Q[j][0] = c;
	    agg_poses->Q[j][1] = -s*a*z[1];
	    agg_poses->Q[j][2] = s*a*z[0];
	    agg_poses->Q[j][3] = 0;
	  }
	  
	}
	else {
	  wavg(agg_poses->Q[j], poses->Q[j], agg_poses->Q[i], w, 4);
	  normalize(agg_poses->Q[j], agg_poses->Q[j], 4);
	}
	agg_poses->W[j] = wtot;
	break;
      }
    }
    if (j == cnt) {  // add a new cluster
      memcpy(agg_poses->X[cnt], poses->X[i], 3*sizeof(double));
      memcpy(agg_poses->Q[cnt], poses->Q[i], 4*sizeof(double));
      agg_poses->W[cnt] = poses->W[i];
      cnt++;
    }
  }
