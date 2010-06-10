/*
 * Copyright (c) 2009 Jared Glover <jglov -=- mit.edu>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: planar_fit.cpp 26020 2009-11-10 23:05:43Z jglov $
 *
 */

/**
@mainpage

@htmlinclude manifest.html

\author Jared Glover

@b Estimate the pose of an object using Bingham pose voting.

 **/



// eigen
#include <Eigen/Core>
#include <Eigen/LU>
USING_PART_OF_NAMESPACE_EIGEN
//#define EIGEN_USE_NEW_STDVECTOR
//#include <Eigen/StdVector>


#include <float.h>

// bingham
extern "C" {
#include <bingham.h>
  //#include <bingham/util.h>
}

// point cloud mapping
//#include <point_cloud_mapping/cloud_io.h>




using namespace std;
//using namespace sensor_msgs;


int getChannel(PointCloud cloud, const char *name)
{
  for (uint i = 0; i < cloud.get_channels_size(); i++)
    if (!cloud.channels[i].name.compare(name))
      return i;

  return -1;
}


/*
 * Returns an array of bingham mixtures
 */
bingham_mix_t *getBinghamMixtures(char *f_bmx, int &k)
{
  FILE *f = fopen(f_bmx, "r");

  if (f == NULL) {
    ROS_ERROR("Invalid filename: %s", f_bmx);
    exit(1);
  }

  // get the number of binghams mixtures in the bmx file
  k = 0;
  char sbuf[1024], *s = sbuf;
  int c;
  while (!feof(f)) {
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d", &c) && c+1 > k)
      k = c+1;
  }
  rewind(f);

  bingham_mix_t *BM = (bingham_mix_t *)calloc(k, sizeof(bingham_mix_t));

  // get the number of binghams in each mixture
  int i;
  while (!feof(f)) {
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d %d", &c, &i) == 2 && i+1 > BM[c].n)
      BM[c].n = i+1;
  }
  rewind(f);

  // allocate space for the binghams
  for (c = 0; c < k; c++) {
    BM[c].w = (double *)calloc(BM[c].n, sizeof(double));
    BM[c].B = (bingham_t *)calloc(BM[c].n, sizeof(bingham_t));
  }

  // read in the binghams and corresponding weights
  int d, j, j2;
  double w;
  int line = 0;
  while (!feof(f)) {
    line++;
    s = sbuf;
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d %d %lf %d", &c, &i, &w, &d) == 4) {
      BM[c].w[i] = w;
      BM[c].B[i].d = d;
      BM[c].B[i].Z = (double *)calloc(d-1, sizeof(double));
      BM[c].B[i].V = new_matrix2(d-1, d);
      s = sword(s, " \t", 5);
      if (sscanf(s, "%lf", &BM[c].B[i].F) < 1)  // read F
	break;
      s = sword(s, " \t", 1);
      for (j = 0; j < d-1; j++) {  // read Z
	if (sscanf(s, "%lf", &BM[c].B[i].Z[j]) < 1)
	  break;
	s = sword(s, " \t", 1);
      }
      if (j < d-1)  // error
	break;
      for (j = 0; j < d-1; j++) {  // read V
	for (j2 = 0; j2 < d; j2++) {
	  if (sscanf(s, "%lf", &BM[c].B[i].V[j][j2]) < 1)
	    break;
	  s = sword(s, " \t", 1);
	}
	if (j2 < d)  // error
	  break;
      }
      if (j < d-1)  // error
	break;
    }
  }
  if (!feof(f)) {  // error
    fprintf(stderr, "Error reading file %s at line %d.\n", f_bmx, line);
    return NULL;
  }
  fclose(f);

  return BM;
}


int *getLabels(PointCloud cloud)
{
  int c = getChannel(cloud, "cluster");

  int n = cloud.get_points_size();
  int *L = (int *)malloc(n*sizeof(int));

  for (int i = 0; i < n; i++)
    L[i] = cloud.channels[c].values[i];

  return L;
}


VectorXd *getPFH(PointCloud cloud, int index)
{
  // find the channel of the first feature coordinate
  int c = getChannel(cloud, "f1");
  int n = 33;  //dbug

  VectorXd *pfh = new VectorXd(n);

  for (int i = 0; i < n; i++)
    (*pfh)[i] = cloud.channels[c+i].values[index];

  return pfh;
}


// r = q1*q2
void quaternionMult(double *r, double *q1, double *q2)
{
  double a1 = q1[0];
  double b1 = q1[1];
  double c1 = q1[2];
  double d1 = q1[3];
  double a2 = q2[0];
  double b2 = q2[1];
  double c2 = q2[2];
  double d2 = q2[3];

  r[0] = a1*a2 - b1*b2 - c1*c2 - d1*d2;
  r[1] = a1*b2 + b1*a2 + c1*d2 - d1*c2;
  r[2] = a1*c2 - b1*d2 + c1*a2 + d1*b2;
  r[3] = a1*d2 + b1*c2 - c1*b2 + d1*a2;
}


Matrix3d quaternionToRotationMatrix(Vector4d q)
{
  Matrix3d R;

  double a = q[0];
  double b = q[1];
  double c = q[2];
  double d = q[3];

  R(0,0) = a*a + b*b - c*c - d*d;
  R(0,1) = 2*b*c - 2*a*d;
  R(0,2) = 2*b*d + 2*a*c;
  R(1,0) = 2*b*c + 2*a*d;
  R(1,1) = a*a - b*b + c*c - d*d;
  R(1,2) = 2*c*d - 2*a*b;
  R(2,0) = 2*b*d - 2*a*c;
  R(2,1) = 2*c*d + 2*a*b;
  R(2,2) = a*a - b*b - c*c + d*d;

  return R;
}


Vector4d rotationMatrixToQuaternion(Matrix3d R)
{
  Vector4d Q;

  double S;
  double tr = R(0,0) + R(1,1) + R(2,2);
  if (tr > 0) {
    S = sqrt(tr+1.0) * 2;  // S=4*qw
    Q[0] = 0.25 * S;
    Q[1] = (R(2,1) - R(1,2)) / S;
    Q[2] = (R(0,2) - R(2,0)) / S;
    Q[3] = (R(1,0) - R(0,1)) / S;
  }
  else if ((R(0,0) > R(1,1)) && (R(0,0) > R(2,2))) {
    S = sqrt(1.0 + R(0,0) - R(1,1) - R(2,2)) * 2;  // S=4*qx 
    Q[0] = (R(2,1) - R(1,2)) / S;
    Q[1] = 0.25 * S;
    Q[2] = (R(0,1) + R(1,0)) / S; 
    Q[3] = (R(0,2) + R(2,0)) / S; 
  }
  else if (R(1,1) > R(2,2)) {
    S = sqrt(1.0 + R(1,1) - R(0,0) - R(2,2)) * 2;  // S=4*qy
    Q[0] = (R(0,2) - R(2,0)) / S;
    Q[1] = (R(0,1) + R(1,0)) / S; 
    Q[2] = 0.25 * S;
    Q[3] = (R(1,2) + R(2,1)) / S; 
  }
  else {
    S = sqrt(1.0 + R(2,2) - R(0,0) - R(1,1)) * 2;  // S=4*qz
    Q[0] = (R(1,0) - R(0,1)) / S;
    Q[1] = (R(0,2) + R(2,0)) / S;
    Q[2] = (R(1,2) + R(2,1)) / S;
    Q[3] = 0.25 * S;
  }

  return Q;
}


/*
 * Returns a stacked array of quaternion pairs, (Q[2*i], Q[2*i+1]).
 */
Vector4d *getQuaternions(PointCloud cloud)
{
  int n = cloud.get_points_size();

  int ch_nx = getChannel(cloud, "nx");
  int ch_ny = getChannel(cloud, "ny");
  int ch_nz = getChannel(cloud, "nz");
  int ch_pcx = getChannel(cloud, "pcx");
  int ch_pcy = getChannel(cloud, "pcy");
  int ch_pcz = getChannel(cloud, "pcz");

  //printf("channels: nx=%d, ny=%d, nz=%d, pcx=%d, pcy=%d, pcz=%d\n", ch_nx, ch_ny, ch_nz, ch_pcx, ch_pcy, ch_pcz);

  Vector4d *Q = (Vector4d *)malloc(2*n*sizeof(Vector4d));

  for (int i = 0; i < n; i++) {
    Matrix3d R;
    double nx = R(0,0) = cloud.channels[ch_nx].values[i];
    double ny = R(1,0) = cloud.channels[ch_ny].values[i];
    double nz = R(2,0) = cloud.channels[ch_nz].values[i];
    double pcx = R(0,1) = cloud.channels[ch_pcx].values[i];
    double pcy = R(1,1) = cloud.channels[ch_pcy].values[i];
    double pcz = R(2,1) = cloud.channels[ch_pcz].values[i];
    R(0,2) = ny*pcz - nz*pcy;
    R(1,2) = nz*pcx - nx*pcz;
    R(2,2) = nx*pcy - ny*pcx;

    Q[2*i] = rotationMatrixToQuaternion(R);

    // get the flipped quaternion
    R(0,1) = -R(0,1);
    R(1,1) = -R(1,1);
    R(2,1) = -R(2,1);
    R(0,2) = -R(0,2);
    R(1,2) = -R(1,2);
    R(2,2) = -R(2,2);

    Q[2*i+1] = rotationMatrixToQuaternion(R);

    //if (L[i] == 0) {
    //  printf("N[%d] = (%f, %f, %f), ", i, nx, ny, nz);
    //  printf("PC[%d] = (%f, %f, %f), ", i, pcx, pcy, pcz);
    //  printf("R3[%d] = (%f, %f, %f), ", i, R(0,2), R(1,2), R(2,2));
    //  printf("Q[%d] = (%f, %f, %f, %f)\n", i, Q[i][0], Q[i][1], Q[i][2], Q[i][3]);
    //  printf("dot(N,PC) = %f\n", nx*pcx + ny*pcy + nz*pcz);
    //}
  }

  return Q;
}


int *getClusterCounts(PointCloud cloud, int *L, int k)
{
  int *npoints = (int *)calloc(k, sizeof(int));

  for (uint i = 0; i < cloud.get_points_size(); i++)
    npoints[L[i]]++;

  return npoints;
}


VectorXd *getClusterMeans(PointCloud cloud, int *L, int k)
{
  int n = cloud.get_points_size();
  VectorXd *F[n];
  for (int i = 0; i < n; i++)
    F[i] = getPFH(cloud, i);

  const int p = F[0]->size();  // all vectors have the same length
  const VectorXd zeros = VectorXd::Zero(p);

  VectorXd *M = new VectorXd[k];

  for (int c = 0; c < k; c++) {
    M[c] = zeros;
    int npoints = 0;
    for (int i = 0; i < n; i++) {
      if (L[i] == c) {
	M[c] += *(F[i]);
	npoints++;
      }
    }
    M[c] /= (double)npoints;
    printf("npoints[%d] = %d\n", c, npoints);
  }

  return M;
}


void getClusterVariances(double *V, PointCloud cloud, VectorXd *M, int *L, int k)
{
  int n = cloud.get_points_size();
  VectorXd *F[n];
  for (int i = 0; i < n; i++)
    F[i] = getPFH(cloud, i);

  const int p = F[0]->size();  // all vectors have the same length
  const VectorXd zeros = VectorXd::Zero(p);

  int npoints[k];
  for (int c = 0; c < k; c++) {
    V[c] = 0;
    npoints[c] = 0;
  }

  for (int i = 0; i < n; i++) {
    int c = L[i];
    V[c] += (M[c] - *(F[i])).squaredNorm();
    npoints[c]++;
  }
  for (int c = 0; c < k; c++)
    V[c] /= (double)npoints[c];
}


// rotate a bingham by a quaternion
void rotateBingham3d(bingham_t *B, double *q)
{
  int d = B->d;
  double v0[4], v1[4], v2[4];
  quaternionMult(v0, q, B->V[0]);
  quaternionMult(v1, q, B->V[1]);
  quaternionMult(v2, q, B->V[2]);

  memcpy(B->V[0], v0, d*sizeof(double));
  memcpy(B->V[1], v1, d*sizeof(double));
  memcpy(B->V[2], v2, d*sizeof(double));
}

// invert and rotate a bingham by a quaternion
void invertAndRotateBingham3d(bingham_t *B, double *q)
{
  double v0[4] = {B->V[0][0], -B->V[0][1], -B->V[0][2], -B->V[0][3]};
  double v1[4] = {B->V[1][0], -B->V[1][1], -B->V[1][2], -B->V[1][3]};
  double v2[4] = {B->V[2][0], -B->V[2][1], -B->V[2][2], -B->V[2][3]};

  quaternionMult(B->V[0], q, v0);
  quaternionMult(B->V[1], q, v1);
  quaternionMult(B->V[2], q, v2);
}


void poseVoting(double **R, int nr, VectorXd **F, Vector4d *Q, double *W, int n,
		VectorXd *M, double *V, bingham_mix_t *BM, int k) //, int *idx)
{
  const int ridge_search = 0;
  const double weight_thresh = .1;

  // get the total number of components in the array of bingham mixtures
  int num_components = 0;
  for (int c = 0; c < k; c++)
    num_components += BM[c].n;

  bingham_mix_t BM_orig;
  BM_orig.n = num_components;
  BM_orig.B = (bingham_t *)calloc(num_components, sizeof(bingham_t));
  BM_orig.w = (double *)calloc(num_components, sizeof(double));
  int cnt = 0;
  for (int c = 0; c < k; c++) {
    for (int j = 0; j < BM[c].n; j++) {
      BM_orig.w[cnt] = BM[c].w[j];
      bingham_copy(&BM_orig.B[cnt], &BM[c].B[j]);
      cnt++;
    }
  }

  bingham_mix_t BM_joint;

  // choose np random points
  int np = 2;
  int idx[np]; //idx[0] = 1735; idx[1] = 2577;  //dbug
  for (int i = 0; i < np; i++)
    idx[i] = floor(frand()*(n-1));

  double q[4];
  int first = 1;
  for (int ii = 0; ii < np; ii++) {

    int i = idx[ii];

    q[0] = Q[i][0];
    q[1] = Q[i][1];
    q[2] = Q[i][2];
    q[3] = Q[i][3];

    // compute PFH likelihoods
    double p_pfh[k];
    int cmax = 0;
    double pmax = 0;
    for (int c = 0; c < k; c++) {
      double d_pfh = (M[c] - *(F[i])).squaredNorm();
      p_pfh[c] = normpdf(d_pfh, 0, 1000); //sqrt(V[c]));
      if (W[c]*p_pfh[c] > pmax) {
	pmax = W[c]*p_pfh[c];
	cmax = c;
      }
    }

    //printf("\n*****  max prob. cluster = %d  *****\n\n", cmax+1);

    //if (cmax != 1 && cmax != 2 && cmax != 4) // && cmax != 5) // && cmax != 9)
    //  continue;


    bingham_mix_t BM_rot;
    bingham_mixture_copy(&BM_rot, &BM_orig);

    // invert and rotate bingham mixtures; then weight bingham mixture
    // components by PFH likelihoods and cluster weights
    cnt = 0;
    for (int c = 0; c < k; c++) {
      for (int j = 0; j < BM[c].n; j++) {
	BM_rot.w[cnt] = p_pfh[c] * BM[c].w[j] * W[c];
	cnt++;
      }
    }

    // normalize mixture weights
    mult(BM_rot.w, BM_rot.w, 1/sum(BM_rot.w, BM_rot.n), BM_rot.n);

    double max_weight = max(BM_rot.w, BM_rot.n);

    cnt = 0;
    for (int c = 0; c < k; c++) {
      for (int j = 0; j < BM[c].n; j++) {
	// don't rotate low-weight binghams
	if (BM_rot.w[cnt] >= weight_thresh*max_weight)
	  invertAndRotateBingham3d(&BM_rot.B[cnt], q);
	cnt++;
      }
    }

    // discard mixture components with low weight
    bingham_mixture_thresh_weights(&BM_rot, weight_thresh*max_weight);


    // multiply new mixture into joint posterior
    if (first) {
      bingham_mixture_copy(&BM_joint, &BM_rot);
      first = 0;
    }
    else {
      
      // addition
      //bingham_mixture_add(&BM_joint, &BM_rot);

      // multiplication
      bingham_mix_t BM_tmp;
      bingham_mixture_mult(&BM_tmp, &BM_joint, &BM_rot);
      bingham_mixture_free(&BM_joint);
      bingham_mixture_copy(&BM_joint, &BM_tmp);     // BM_joint *= BM_rot
      bingham_mixture_free(&BM_tmp);

      // weight thresholding
      max_weight = max(BM_joint.w, BM_joint.n);
      bingham_mixture_thresh_weights(&BM_joint, weight_thresh*max_weight);
    }

    /*
    for (int j = 0; j < BM_joint.n; j++) {
      bingham_t *B = &BM_joint.B[j];
      printf("B[%d]->F = %f\n", j, B->F);
      printf("B[%d]->Z = [%f %f %f]\n", j, B->Z[0], B->Z[1], B->Z[2]);
      printf("B[%d]->V[0] = [%f %f %f %f]\n", j, B->V[0][0], B->V[0][1], B->V[0][2], B->V[0][3]);
      printf("B[%d]->V[1] = [%f %f %f %f]\n", j, B->V[1][0], B->V[1][1], B->V[1][2], B->V[1][3]);
      printf("B[%d]->V[2] = [%f %f %f %f]\n", j, B->V[2][0], B->V[2][1], B->V[2][2], B->V[2][3]);
    }
    printf("\n\n");
    */

    //printf("V = [ ");
    //for (int c = 0; c < k; c++)
    //  printf("%f ", V[c]);
    //printf("]\n");

    /* dbug
    double psum = 0;
    for (int c = 0; c < k; c++)
      psum += W[c]*p_pfh[c];
    printf("W * p_pfh = [ ");
    for (int c = 0; c < k; c++)
      printf("%f ", W[c]*p_pfh[c]/psum);
    printf("]\n\n");
    printf("w = [ ");
    for (int j = 0; j < BM_rot.n; j++)
      printf("%f ", BM_rot.w[j]);
    printf("]\n");
    printf("\n");

    for (int j = 0; j < BM_rot.n; j++) {
      bingham_t *B = &BM_rot.B[j];
      printf("B[%d]->F = %f\n", j, B->F);
      printf("B[%d]->Z = [%f %f %f]\n", j, B->Z[0], B->Z[1], B->Z[2]);
      printf("B[%d]->V[0] = [%f %f %f %f]\n", j, B->V[0][0], B->V[0][1], B->V[0][2], B->V[0][3]);
      printf("B[%d]->V[1] = [%f %f %f %f]\n", j, B->V[1][0], B->V[1][1], B->V[1][2], B->V[1][3]);
      printf("B[%d]->V[2] = [%f %f %f %f]\n", j, B->V[2][0], B->V[2][1], B->V[2][2], B->V[2][3]);
    }
    printf("\n\n");
    */

    int nsamples = 30;

    /*
    //max_peak = bingham_mixture_peak(&BM_rot);
    double **X = new_matrix2(nsamples, 4);
    bingham_mixture_sample_ridge(X, &BM_rot, nsamples, 1/surface_area_sphere(3)); //.1*max_peak);
    printf("X = [ ...\n");
    for (int j = 0; j < nsamples; j++)
      printf("%f, %f, %f, %f ; ...\n", X[j][0], X[j][1], X[j][2], X[j][3]);
    printf("];\n\n");
    free_matrix2(X);
    */

    /*
    int *indices = (int *)calloc(nr, sizeof(int));
    double *pdf = (double *)calloc(nr, sizeof(double));
    for (int i = 0; i < nr; i++)
      pdf[i] = -bingham_mixture_pdf(R[i], &BM_rot);
    mult(pdf, pdf, -1/min(pdf, nr), nr);  //dbug
    sort_indices(pdf, indices, nr);
    printf("X = [ ...\n");
    for (int j = 0; j < nsamples; j++)
      printf("%f, %f, %f, %f ; ...\n", R[indices[j]][0], R[indices[j]][1], R[indices[j]][2], R[indices[j]][3]);
    printf("];\n\n");
    printf("pdf = [ ");
    for (int j = 0; j < nsamples; j++)
      printf("%f ", -pdf[indices[j]]);
    printf("]\n\n");
    free(indices);
    free(pdf);
    */

    bingham_mixture_free(&BM_rot);
  }

  /*
  printf("w = [ ");
  for (int j = 0; j < BM_joint.n; j++)
    printf("%f ", BM_joint.w[j]);
  printf("]\n");

  printf("\n");
  printf("BM_joint.n = %d\n", BM_joint.n);

  for (int j = 0; j < BM_joint.n; j++) {
    bingham_t *B = &BM_joint.B[j];
    printf("B[%d]->F = %f\n", j, B->F);
    printf("B[%d]->Z = [%f %f %f]\n", j, B->Z[0], B->Z[1], B->Z[2]);
    printf("B[%d]->V[0] = [%f %f %f %f]\n", j, B->V[0][0], B->V[0][1], B->V[0][2], B->V[0][3]);
    printf("B[%d]->V[1] = [%f %f %f %f]\n", j, B->V[1][0], B->V[1][1], B->V[1][2], B->V[1][3]);
    printf("B[%d]->V[2] = [%f %f %f %f]\n", j, B->V[2][0], B->V[2][1], B->V[2][2], B->V[2][3]);
  }

  printf("\n\n");

  printf("idx = [ ");
  for (int i = 0; i < np; i++)
    printf("%d ", idx[i]+1);
  printf("]\n\n");
  */

  int nsamples = 30;
  //double max_peak = bingham_mixture_peak(&BM_joint);


  if (ridge_search) {

    //---------- ridge search ----------//

    double **X = new_matrix2(nsamples, 4);
    bingham_mixture_sample_ridge(X, &BM_joint, nsamples, 1/surface_area_sphere(3)); //.1*max_peak);
    printf("X = [ ...\n");
    for (int j = 0; j < nsamples; j++)
      printf("%f, %f, %f, %f ; ...\n", X[j][0], X[j][1], X[j][2], X[j][3]);
    printf("];\n\n");
    free_matrix2(X);

  }
  /*
  else {

    //---------- tessellation search ----------//

    int *indices = (int *)calloc(nr, sizeof(int));
    double *pdf = (double *)calloc(nr, sizeof(double));
    for (int i = 0; i < nr; i++)
      pdf[i] = -bingham_mixture_pdf(R[i], &BM_joint);
    mult(pdf, pdf, -1/min(pdf, nr), nr);  //dbug
    sort_indices(pdf, indices, nr);
    printf("X = [ ...\n");
    for (int j = 0; j < nsamples; j++)
      printf("%f, %f, %f, %f ; ...\n", R[indices[j]][0], R[indices[j]][1], R[indices[j]][2], R[indices[j]][3]);
    printf("];\n\n");
    printf("pdf = [ ");
    for (int j = 0; j < nsamples; j++)
      printf("%f ", -pdf[indices[j]]);
    printf("]\n\n");
    free(indices);
    free(pdf);
  }
  */

  bingham_mixture_free(&BM_orig);
}


void estimatePose(char *f_bmx, PointCloud model_cloud, PointCloud observed_cloud)
{
  // initialize bingham library
  ros::Time ts = ros::Time::now();
  //bingham_init();
  ROS_DEBUG("Initialized Bingham library in %f seconds.", (ros::Time::now() - ts).toSec());

  int n = model_cloud.get_points_size();

  int *L = getLabels(model_cloud);

  // get the number of clusters, k
  int k = 1;
  for (int i = 0; i < n; i++)
    if (L[i] + 1 > k)
      k = L[i] + 1;

  // get cluster means, variances, and weights
  VectorXd *M = getClusterMeans(model_cloud, L, k);
  double V[k];
  getClusterVariances(V, model_cloud, M, L, k);
  double W[k];
  int *C = getClusterCounts(model_cloud, L, k);
  int csum = 0;
  for (int i = 0; i < k; i++)
    csum += C[i];
  for (int i = 0; i < k; i++)
    W[i] = C[i]/(double)csum;

  // get binghams
  int num_bmix;
  bingham_mix_t *BM = getBinghamMixtures(f_bmx, num_bmix);

  if (num_bmix != k) {
    ROS_ERROR("k != num_bmix...quitting");
    exit(1);
  }

  // discretize the space of quaternions
  bingham_pmf_t pmf;
  //bingham_discretize(&pmf, &BM[0].B[0], 10000);
  double **R = NULL; pmf.points;
  int nr = 0;
  /*
  for (int i = 0; i < pmf.n; i++) {  // only need to keep one hemisphere of S3
    if (R[i][0] >= 0) {
      R[nr][0] = R[i][0];
      R[nr][1] = R[i][1];
      R[nr][2] = R[i][2];
      R[nr][3] = R[i][3];
      nr++;
    }
  }

  //dbug
  R[0][0] = 0;
  R[0][1] = 0;
  R[0][2] = 0;
  R[0][3] = 1;

  printf("nr = %d\n", nr);
  */

  // vote for pose
  n = observed_cloud.get_points_size();
  VectorXd *F[n];
  for (int i = 0; i < n; i++)
    F[i] = getPFH(observed_cloud, i);
  Vector4d *Q = getQuaternions(observed_cloud);

  ts = ros::Time::now();

  for (int i = 0; i < 100000; i++)  //dbug
    poseVoting(R, nr, F, Q, W, n, M, V, BM, k); //, &i);

  ROS_INFO("Estimated orientation in %f seconds.", (ros::Time::now() - ts).toSec());
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_model> <bmx_model> <pcd_observed>", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 4)
    usage(argc, argv);

  char *f_model = argv[1];
  char *f_bmx = argv[2];
  char *f_obs = argv[3];
  PointCloud model_cloud;
  PointCloud observed_cloud;
  cloud_io::loadPCDFile(f_model, model_cloud);
  cloud_io::loadPCDFile(f_obs, observed_cloud);

  ROS_DEBUG("Loaded PCD model '%s' with %d points and %d channels.",
	   f_model, model_cloud.get_points_size(), model_cloud.get_channels_size());

  estimatePose(f_bmx, model_cloud, observed_cloud);

  return 0;
}
