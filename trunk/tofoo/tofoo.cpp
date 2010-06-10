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

@b Estimate the orientation of an object using TOFOO Bagging.

 **/



// eigen
#include <Eigen/Core>
#include <Eigen/LU>
USING_PART_OF_NAMESPACE_EIGEN
#define EIGEN_USE_NEW_STDVECTOR
//#include <Eigen/StdVector>


#include <float.h>

// bingham
extern "C" {
#include <bingham.h>
#include <bingham/bingham_constants.h>
}

// point cloud mapping
//#include <point_cloud_mapping/cloud_io.h>
#include "pointcloud.h"

#define ROS_INFO printf
#define ROS_ERROR printf
#define ROS_DEBUG printf




using namespace std;
//using namespace sensor_msgs;



typedef struct {
  VectorXd *M;
  double *V;
  double *W;
  bingham_mix_t *BM;
  int k;
} tofoo_model_t;




int getChannel(PointCloud cloud, const char *name)
{
  for (uint i = 0; i < cloud.get_channels_size(); i++)
    if (!cloud.channels[i].name.compare(name))
      return i;

  return -1;
}


/*
 * Returns an array of bingham mixtures
 *
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
*/


/*
 * Write bingham mixtures in the following format:
 * c:  cluster
 * i:  mixture component
 * w:  component weight
 * d:  dimension (4 for quaternions)
 * F:  normalization constant (from lookup table)
 * Z:  (lambda) concentration params
 * V:  direction params
 *
 * B <c> <i> <w> <d> <F> <Z> <V>
 *
void writeBinghamMixtures(bingham_mix_t *BM, int num_clusters, char *fout)
{
  printf("saving BMM to %s\n", fout);

  FILE *f = fopen(fout, "w");
  int c, i, j, k;

  for (c = 0; c < num_clusters; c++) {
    for (i = 0; i < BM[c].n; i++) {

      double w = BM[c].w[i];
      int d = BM[c].B[i].d;
      double F = BM[c].B[i].F;
      double *Z = BM[c].B[i].Z;
      double **V = BM[c].B[i].V;

      fprintf(f, "B %d %d %f ", c, i, w);
      fprintf(f, "%d %f ", d, F);
      for (j = 0; j < d-1; j++)
	fprintf(f, "%f ", Z[j]);
      for (j = 0; j < d-1; j++)
	for (k = 0; k < d; k++)
	  fprintf(f, "%f ", V[j][k]);
      fprintf(f, "\n");
    }
  }

  fclose(f);
}
*/


int *getLabels(PointCloud cloud)
{
  int c = getChannel(cloud, "cluster");

  int n = cloud.get_points_size();
  int *L = (int *)malloc(n*sizeof(int));

  for (int i = 0; i < n; i++)
    L[i] = (int)cloud.channels[c].values[i];

  return L;
}


VectorXd *getPFH(PointCloud cloud, int ch1, int index)
{
  // find the channel of the first feature coordinate
  int c = ch1;
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

  //Vector4d *Q = (Vector4d *)malloc(2*n*sizeof(Vector4d));
  Vector4d *Q = (Vector4d *)malloc(n*sizeof(Vector4d));

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

    //Q[2*i] = rotationMatrixToQuaternion(R);
    Q[i] = rotationMatrixToQuaternion(R);

    // get the flipped quaternion
    //R(0,1) = -R(0,1);
    //R(1,1) = -R(1,1);
    //R(2,1) = -R(2,1);
    //R(0,2) = -R(0,2);
    //R(1,2) = -R(1,2);
    //R(2,2) = -R(2,2);

    //Q[2*i+1] = rotationMatrixToQuaternion(R);

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
  int ch1 = getChannel(cloud, "f1");
  for (int i = 0; i < n; i++)
    F[i] = getPFH(cloud, ch1, i);

  const int p = F[0]->size();  // all vectors have the same length
  const VectorXd zeros = VectorXd::Zero(p);

  VectorXd *M = new VectorXd[k];

  printf("npoints = [ ");
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
    printf("%d  ", npoints);
  }
  printf("]\n");

  return M;
}


void getClusterVariances(double *V, PointCloud cloud, VectorXd *M, int *L, int k)
{
  int n = cloud.get_points_size();
  VectorXd *F[n];
  int ch1 = getChannel(cloud, "f1");
  for (int i = 0; i < n; i++)
    F[i] = getPFH(cloud, ch1, i);

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


// compress a Bingham mixture, in place
void compressBMM(bingham_mix_t *BM, double compression_thresh, int start_idx)
{
  int n = BM->n;

  bingham_stats_t stats[n];
  for (int i = 0; i < n; i++)
    bingham_stats(&stats[i], &BM->B[i]);

  while (1) {

    int num_compressed = 0;

    for (int i = 0; i < n; i++) {

      // compute the min KL divergence from B[i] to every B[j] (for j>i)
      double dmin = DBL_MAX;
      int jmin = 0;
      int j0 = MAX(start_idx, i+1);
      for (int j = j0; j < n; j++) {
	double d = bingham_KL_divergence(&stats[i], &stats[j]);
	//printf("d_KL[%d][%d] = %f\n", i, j, d);
	if (d < dmin) {
	  dmin = d;
	  jmin = j;
	}
      }

      if (dmin < compression_thresh) {

	//printf("***** compressing %d and %d with d_KL = %f *****\n", i, jmin, dmin);

	// merge B[i] and B[j]
	int j = jmin;
	bingham_t tmp;
	double alpha = BM->w[i] / (BM->w[i] + BM->w[j]);
	bingham_merge(&tmp, &stats[i], &stats[j], alpha);

	// free the old structs
	bingham_free(&BM->B[i]);
	bingham_free(&BM->B[j]);
	bingham_stats_free(&stats[i]);
	bingham_stats_free(&stats[j]);

	// copy tmp --> B[i]
	memcpy(&BM->B[i], &tmp, sizeof(bingham_t));
	BM->w[i] += BM->w[j];
	bingham_stats(&stats[i], &BM->B[i]);

	// copy B[n-1] --> B[j]
	if (j < n-1) {
	  memcpy(&BM->B[j], &BM->B[n-1], sizeof(bingham_t));
	  BM->w[j] = BM->w[n-1];
	  memcpy(&stats[j], &stats[n-1], sizeof(bingham_stats_t));
	  stats[j].B = &BM->B[j];  // repair ptr
	}

	// decrement n
	n--;

	num_compressed++;
      }
    }

    //if (num_compressed == 0)
      break;
  }

  for (int i = 0; i < n; i++)
    bingham_stats_free(&stats[i]);

  //printf(" --> Compressed BMM from %d to %d\n", BM->n, n);

  BM->n = n;
}



//------------------- TOFOO Bagging --------------------//


// parameters
int bag_size = 1;  //50 ;              // number of samples to aggregate
int sample_size = 1;  //2;             // points per sample
double weight_thresh = .0001;  //.05;      // BMM component weight threshold
int compress = 0;  //1;                // whether or not to do mixture compression
double compression_thresh = 20;   // KL divergence threshold for BMM compression
double zcap = -10000000;  //-200;              // minimum concentration
int use_entropy_weights = 0;
int hard_assignment = 1;
int point_idx = -1;


/*
 * Bagged TOFOO
 */
bingham_mix_t *tofooBagging(VectorXd **F, Vector4d *Q, int n, tofoo_model_t *T)
{
  VectorXd *M = T->M;
  //double *V = T->V;
  double *W = T->W;
  bingham_mix_t *BM = T->BM;
  int k = T->k;

  // get the total number of components in the array of bingham mixtures
  int num_components = 0;
  for (int c = 0; c < k; c++)
    num_components += BM[c].n;

  // compute the average negative exponential entropy of BMMs
  double entropy_weights[k];

  if (use_entropy_weights) {
    for (int c = 0; c < k; c++) {
      entropy_weights[c] = 0;
      for (int j = 0; j < BM[c].n; j++) {
	bingham_stats_t stats;
	bingham_stats(&stats, &BM[c].B[j]);
	entropy_weights[c] += BM[c].w[j] * stats.entropy;
      }
      entropy_weights[c] = exp(-entropy_weights[c]);
    }
    mult(entropy_weights, entropy_weights, 1/max(entropy_weights, k), k);
    //for (int c = 0; c < k; c++)
    //  printf("entropy_weights[%d] = %f\n", c+1, entropy_weights[c]);
  }



  // flatten out T->BM into a single mixture
  bingham_mix_t BM_orig;
  BM_orig.n = num_components;
  BM_orig.B = (bingham_t *)calloc(num_components, sizeof(bingham_t));
  BM_orig.w = (double *)calloc(num_components, sizeof(double));
  int cnt = 0;
  for (int c = 0; c < k; c++) {
    for (int j = 0; j < BM[c].n; j++) {
      BM_orig.w[cnt] = BM[c].w[j];
      bingham_copy(&BM_orig.B[cnt], &BM[c].B[j]);
      //printf("BM_orig.B[%d].Z = (%f, %f, %f)\n", cnt, BM_orig.B[cnt].Z[0], BM_orig.B[cnt].Z[1], BM_orig.B[cnt].Z[2]);  //dbug
      //printf("BM[%d].B[%d].Z = (%f, %f, %f)\n", c, j, BM[c].B[j].Z[0], BM[c].B[j].Z[1], BM[c].B[j].Z[2]);  //dbug
      cnt++;
    }
  }

  bingham_mix_t *BM_joint = (bingham_mix_t *)calloc(1, sizeof(bingham_mix_t));
  int idx[sample_size];

  //printf("idx = [];\n");

  for (int sample = 0; sample < bag_size; sample++) {

    bingham_mix_t BM_sample;

    //printf("\n");

    double q[4];
    int first = 1;
    for (int ii = 0; ii < sample_size; ii++) {

      double p_pfh[k];
      int cmax;
      double pmax;
      int i;

      while (1) {
	// sample a random point
	idx[ii] = (int)floor(frand()*(n-1));

	if (ii==0 && point_idx > 0)
	  idx[ii] = point_idx - 1;

	i = idx[ii];

	q[0] = Q[i][0];
	q[1] = Q[i][1];
	q[2] = Q[i][2];
	q[3] = Q[i][3];

	// compute PFH likelihoods
	cmax = 0;
	pmax = 0;
	for (int c = 0; c < k; c++) {
	  double d_pfh = (M[c] - *(F[i])).squaredNorm();
	  p_pfh[c] = normpdf(d_pfh, 0, 1000); //sqrt(V[c]));
	  if (W[c]*p_pfh[c] > pmax) {
	    pmax = W[c]*p_pfh[c];
	    cmax = c;
	  }
	}

	printf("idx[%d] = %d\n", ii, i+1);
	printf("*****  max prob. cluster = %d  *****\n", cmax+1);

	//if (cmax+1 == 6) {  //dbug!!!

	// accept sample with probability entropy_weights[cmax]
	if (!use_entropy_weights)
	  break;
	if (frand() <= entropy_weights[cmax]) {
	  //printf(" --> ACCEPTED\n\n");
	  break;
	}

	//}

	//printf(" --> rejected\n");
      }

      bingham_mix_t BM_rot;
      bingham_mixture_copy(&BM_rot, &BM_orig);

      // invert and rotate bingham mixtures; then weight bingham mixture
      // components by PFH likelihoods and cluster weights
      cnt = 0;
      for (int c = 0; c < k; c++) {
	for (int j = 0; j < BM[c].n; j++) {
	  BM_rot.w[cnt] = p_pfh[c] * BM[c].w[j] * W[c];
	  if (hard_assignment && c != cmax)
	    BM_rot.w[cnt] = 0;
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
	  if (BM_rot.w[cnt] >= weight_thresh*max_weight) {
	    invertAndRotateBingham3d(&BM_rot.B[cnt], q);
	    //printf("BM_rot.B[%d].Z = (%f, %f, %f)\n", cnt, BM_rot.B[cnt].Z[0], BM_rot.B[cnt].Z[1], BM_rot.B[cnt].Z[2]);  //dbug
	  }
	  cnt++;
	}
      }

      // discard mixture components with low weight
      if (weight_thresh > 0)
	bingham_mixture_thresh_weights(&BM_rot, weight_thresh*max_weight);

      // multiply new mixture into sample posterior
      if (first) {
	bingham_mixture_copy(&BM_sample, &BM_rot);
	first = 0;
      }
      else {
	// multiplication:  BM_sample *= BM_rot
	bingham_mix_t BM_tmp;
	bingham_mixture_mult(&BM_tmp, &BM_sample, &BM_rot);
	bingham_mixture_free(&BM_sample);
	memcpy(&BM_sample, &BM_tmp, sizeof(bingham_mix_t));

	// weight thresholding
	max_weight = max(BM_sample.w, BM_sample.n);
	if (weight_thresh > 0)
	  bingham_mixture_thresh_weights(&BM_sample, weight_thresh*max_weight);

	// concentration capping
	int d = BM_sample.B[0].d;
	for (int j = 0; j < BM_sample.n; j++) {
	  int capped = 0;
	  for (int z = 0; z < d-1; z++) {
	    if (BM_sample.B[j].Z[z] < zcap) {
	      BM_sample.B[0].Z[z] = zcap;
	      capped = 1;
	    }
	  }
	  if (capped)
	    BM_sample.B[j].F = bingham_F_lookup_3d(BM_sample.B[j].Z);
	}
      }

      bingham_mixture_free(&BM_rot);
    }

    // add sample to the joint posterior
    if (sample == 0)
      bingham_mixture_copy(BM_joint, &BM_sample);
    else
      bingham_mixture_add(BM_joint, &BM_sample);

    // BMM compression
    //if (compress)
    //  compressBMM(BM_joint, compression_thresh, BM_joint->n - BM_sample.n);

    // weight thresholding
    double max_weight = max(BM_joint->w, BM_joint->n);
    if (weight_thresh > 0)
      bingham_mixture_thresh_weights(BM_joint, weight_thresh*max_weight);

    bingham_mixture_free(&BM_sample);
  }

  bingham_mixture_free(&BM_orig);

  // BMM compression
  //if (compress)
  //  compressBMM(BM_joint, compression_thresh, 0);

  // weight thresholding
  double max_weight = max(BM_joint->w, BM_joint->n);
  if (weight_thresh > 0)
    bingham_mixture_thresh_weights(BM_joint, weight_thresh*max_weight);

  printf(" --> Weight thresholded BMM to %d components\n", BM_joint->n);

  return BM_joint;
}

//------------------- End TOFOO Bagging --------------------//



void estimatePose(char *f_bmx_out, char *f_bmx_in, PointCloud model_cloud, PointCloud observed_cloud)
{
  // initialize bingham library
  //ros::Time ts = ros::Time::now();
  //bingham_init();
  //ROS_DEBUG("Initialized Bingham library in %f seconds.", (ros::Time::now() - ts).toSec());

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

  //ROS_INFO("Loaded PFH models in %f seconds.", (ros::Time::now() - ts).toSec());



  //ts = ros::Time::now();

  // get binghams
  int num_bmix;
  bingham_mix_t *BM = load_bmx(f_bmx_in, &num_bmix);  //getBinghamMixtures(f_bmx_in, num_bmix);

  //ROS_INFO("Loaded BMMs in %f seconds.", (ros::Time::now() - ts).toSec());


  if (num_bmix != k) {
    ROS_ERROR("k != num_bmix...quitting");
    exit(1);
  }

  // create tofoo model
  tofoo_model_t T;
  T.M = M;
  T.V = V;
  T.W = W;
  T.BM = BM;
  T.k = k;

  //ts = ros::Time::now();

  // get observations
  n = observed_cloud.get_points_size();
  VectorXd *F[n];
  int ch1 = getChannel(observed_cloud, "f1");
  for (int i = 0; i < n; i++)
    F[i] = getPFH(observed_cloud, ch1, i);
  Vector4d *Q = getQuaternions(observed_cloud);

  //ROS_INFO("Loaded observations in %f seconds.", (ros::Time::now() - ts).toSec());


  //ts = ros::Time::now();

  // compute orientation posterior
  bingham_mix_t *BMM_out = tofooBagging(F, Q, n, &T);

  //ROS_INFO("Estimated orientation in %f seconds.", (ros::Time::now() - ts).toSec());

  // write bingham mixture posterior to file
  //writeBinghamMixtures(BMM_out, 1, f_bmx_out);
  save_bmx(BMM_out, 1, f_bmx_out);
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_model> <bmx_model> <pcd_observed> <bmx_out> [bag_size] [point_idx]", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 5)
    usage(argc, argv);

  printf("\n\n");

  char *f_model = argv[1];
  char *f_bmx_in = argv[2];
  char *f_obs = argv[3];
  char *f_bmx_out = argv[4];
  if (argc >= 6)
    bag_size = (int)atof(argv[5]);
  if (argc >= 6)
    point_idx = (int)atof(argv[6]);

  //ros::Time ts = ros::Time::now();

  PointCloud model_cloud;
  PointCloud observed_cloud;
  loadPCDFile(f_model, model_cloud);  //cloud_io::loadPCDFile(f_model, model_cloud);
  loadPCDFile(f_obs, observed_cloud);  //cloud_io::loadPCDFile(f_obs, observed_cloud);

  //ROS_INFO("Loaded PCD models %f seconds.", (ros::Time::now() - ts).toSec());

  estimatePose(f_bmx_out, f_bmx_in, model_cloud, observed_cloud);

  return 0;
}
