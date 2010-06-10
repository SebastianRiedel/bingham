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
#include <point_cloud_mapping/cloud_io.h>




using namespace std;
using namespace sensor_msgs;


int getChannel(PointCloud cloud, const char *name)
{
  for (uint i = 0; i < cloud.get_channels_size(); i++)
    if (!cloud.channels[i].name.compare(name))
      return i;

  return -1;
}


/*
 * Returns a stacked array of Bingham pairs, (B[2*i], B[2*i+1]).
 */
bingham_t *getBinghams(PointCloud cloud, int *L, int k)
{
  bingham_t *B = (bingham_t *)calloc(2*k, sizeof(bingham_t));

  int n = cloud.get_points_size();

  int c0 = getChannel(cloud, "b1f");

  for (int j = 0; j < k; j++) {

    B[2*j].d = 4;
    B[2*j].Z = (double *)calloc(3, sizeof(double));
    B[2*j].V = new_matrix2(3,4);
    B[2*j+1].d = 4;
    B[2*j+1].Z = (double *)calloc(3, sizeof(double));
    B[2*j+1].V = new_matrix2(3,4);

    for (int i = 0; i < n; i++) {
      if (L[i] == j) {

	int c = c0;

	B[2*j].F = cloud.channels[c++].values[i];
	B[2*j].Z[0] = cloud.channels[c++].values[i];
	B[2*j].Z[1] = cloud.channels[c++].values[i];
	B[2*j].Z[2] = cloud.channels[c++].values[i];
	B[2*j].V[0][0] = cloud.channels[c++].values[i];
	B[2*j].V[0][1] = cloud.channels[c++].values[i];
	B[2*j].V[0][2] = cloud.channels[c++].values[i];
	B[2*j].V[0][3] = cloud.channels[c++].values[i];
	B[2*j].V[1][0] = cloud.channels[c++].values[i];
	B[2*j].V[1][1] = cloud.channels[c++].values[i];
	B[2*j].V[1][2] = cloud.channels[c++].values[i];
	B[2*j].V[1][3] = cloud.channels[c++].values[i];
	B[2*j].V[2][0] = cloud.channels[c++].values[i];
	B[2*j].V[2][1] = cloud.channels[c++].values[i];
	B[2*j].V[2][2] = cloud.channels[c++].values[i];
	B[2*j].V[2][3] = cloud.channels[c++].values[i];

	B[2*j+1].F = cloud.channels[c++].values[i];
	B[2*j+1].Z[0] = cloud.channels[c++].values[i];
	B[2*j+1].Z[1] = cloud.channels[c++].values[i];
	B[2*j+1].Z[2] = cloud.channels[c++].values[i];
	B[2*j+1].V[0][0] = cloud.channels[c++].values[i];
	B[2*j+1].V[0][1] = cloud.channels[c++].values[i];
	B[2*j+1].V[0][2] = cloud.channels[c++].values[i];
	B[2*j+1].V[0][3] = cloud.channels[c++].values[i];
	B[2*j+1].V[1][0] = cloud.channels[c++].values[i];
	B[2*j+1].V[1][1] = cloud.channels[c++].values[i];
	B[2*j+1].V[1][2] = cloud.channels[c++].values[i];
	B[2*j+1].V[1][3] = cloud.channels[c++].values[i];
	B[2*j+1].V[2][0] = cloud.channels[c++].values[i];
	B[2*j+1].V[2][1] = cloud.channels[c++].values[i];
	B[2*j+1].V[2][2] = cloud.channels[c++].values[i];
	B[2*j+1].V[2][3] = cloud.channels[c++].values[i];

	break;
      }
    }
  }

  return B;
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


Vector4d getRotationBetweenFrames(Vector4d q1, Vector4d q2)
{
  Matrix3d A1 = quaternionToRotationMatrix(q1);
  Matrix3d A2 = quaternionToRotationMatrix(q2);

  MatrixXd A = MatrixXd::Zero(9,9);
  A(0,0) = A1(0,0);  A(0,1) = A1(1,0);  A(0,2) = A1(2,0);
  A(1,3) = A1(0,0);  A(1,4) = A1(1,0);  A(1,5) = A1(2,0);
  A(2,6) = A1(0,0);  A(2,7) = A1(1,0);  A(2,8) = A1(2,0);
  A(3,0) = A1(0,1);  A(3,1) = A1(1,1);  A(3,2) = A1(2,1);
  A(4,3) = A1(0,1);  A(4,4) = A1(1,1);  A(4,5) = A1(2,1);
  A(5,6) = A1(0,1);  A(5,7) = A1(1,1);  A(5,8) = A1(2,1);
  A(6,0) = A1(0,2);  A(6,1) = A1(1,2);  A(6,2) = A1(2,2);
  A(7,3) = A1(0,2);  A(7,4) = A1(1,2);  A(7,5) = A1(2,2);
  A(8,6) = A1(0,2);  A(8,7) = A1(1,2);  A(8,8) = A1(2,2);

  VectorXd b(9);
  b[0] = A2(0,0);  b[1] = A2(1,0);  b[2] = A2(2,0);
  b[3] = A2(0,1);  b[4] = A2(1,1);  b[5] = A2(2,1);
  b[6] = A2(0,2);  b[7] = A2(1,2);  b[8] = A2(2,2);

  VectorXd x;
  A.lu().solve(b, &x);

  Matrix3d R;
  R(0,0) = x[0];  R(0,1) = x[1];  R(0,2) = x[2];
  R(1,0) = x[3];  R(1,1) = x[4];  R(1,2) = x[5];
  R(2,0) = x[6];  R(2,1) = x[7];  R(2,2) = x[8];

  return rotationMatrixToQuaternion(R);
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


VectorXd *getClusterMeans(PointCloud cloud, int *L, int k)
{
  int n = cloud.get_points_size();
  VectorXd *F[n];
  for (int i = 0; i < n; i++)
    F[i] = getPFH(cloud, i);

  const int p = F[0]->size();  // all vectors have the same length
  const VectorXd zeros = VectorXd::Zero(p);

  // initialize centroids M by randomly sampling from X
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
  }

  return M;
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
}


void poseVoting(PointCloud cloud, VectorXd *M, bingham_t *B, int k)
{
  double lambda = .05;

  int n = cloud.get_points_size();
  VectorXd *F[n];
  for (int i = 0; i < n; i++)
    F[i] = getPFH(cloud, i);

  Vector4d *Q = getQuaternions(cloud);

  // discretize the space of quaternions
  bingham_pmf_t pmf;
  bingham_discretize(&pmf, &B[0], 1000);

  // allocate space for rotated binghams
  bingham_t *B_rot = (bingham_t *)malloc(2*k*sizeof(bingham_t));
  for (int c = 0; c < 2*k; c++)
    bingham_copy(&B_rot[c], &B[c]);

  // compute the probability mass of each cell in the pmf

  for (int i = 0; i < pmf.n; i++) {  // cells
    
    //printf("q = [%f, %f, %f, %f];\n\n", pmf.points[i][0], pmf.points[i][1],
    //	   pmf.points[i][2], pmf.points[i][3]);

    // global object coordinate frame
    Vector4d q_obj(pmf.points[i][0], pmf.points[i][1], pmf.points[i][2], pmf.points[i][3]);

    // rotate bingham distributions
    int d = B[0].d;
    for (int c = 0; c < 2*k; c++) {
      memcpy(B_rot[c].V[0], B[c].V[0], d*(d-1)*sizeof(double));
      rotateBingham3d(&B_rot[c], pmf.points[i]);

      /*
      printf("B(%d).F = %f\n", c+1, B[c].F);
      printf("B(%d).Z = [%f %f %f]\n", c+1, B[c].Z[0], B[c].Z[1], B[c].Z[2]);
      printf("B(%d).V = [%f %f %f %f ; %f %f %f %f ; %f %f %f %f]\n\n", c+1,
      	     B[c].V[0][0], B[c].V[0][1], B[c].V[0][2], B[c].V[0][3], 
      	     B[c].V[1][0], B[c].V[1][1], B[c].V[1][2], B[c].V[1][3], 
      	     B[c].V[2][0], B[c].V[2][1], B[c].V[2][2], B[c].V[2][3]);

      printf("B_rot(%d).F = %f\n", c+1, B_rot[c].F);
      printf("B_rot(%d).Z = [%f %f %f]\n", c+1, B_rot[c].Z[0], B_rot[c].Z[1], B_rot[c].Z[2]);
      printf("B_rot(%d).V = [%f %f %f %f ; %f %f %f %f ; %f %f %f %f]\n\n", c+1,
      	     B_rot[c].V[0][0], B_rot[c].V[0][1], B_rot[c].V[0][2], B_rot[c].V[0][3], 
      	     B_rot[c].V[1][0], B_rot[c].V[1][1], B_rot[c].V[1][2], B_rot[c].V[1][3], 
      	     B_rot[c].V[2][0], B_rot[c].V[2][1], B_rot[c].V[2][2], B_rot[c].V[2][3]);
      */
    }

    //sleep(10);

    printf(".");
    fflush(0);

    pmf.mass[i] = 0;
    for (int j = 2; j < n; j++) {  // points

      /* compute the rotations that take the global frame of cell i, q_obj,
	 into the local frames at point j, Q[2*j] and Q[2*j+1]  */

      //Vector4d q1vec = getRotationBetweenFrames(q_obj, Q[2*j]);
      //Vector4d q2vec = getRotationBetweenFrames(q_obj, Q[2*j+1]);

      Vector4d q1vec = Q[2*j];  //getRotationBetweenFrames(q_obj, Q[2*j]);
      Vector4d q2vec = Q[2*j+1];  //getRotationBetweenFrames(q_obj, Q[2*j+1]);
      double q1[4], q2[4];
      q1[0] = q1vec[0]; q1[1] = q1vec[1]; q1[2] = q1vec[2]; q1[3] = q1vec[3];
      q2[0] = q2vec[0]; q2[1] = q2vec[1]; q2[2] = q2vec[2]; q2[3] = q2vec[3];

      //printf("q1 = [%f, %f, %f, %f];\n\n", q1[0], q1[1], q1[2], q1[3]);
      //printf("q2 = [%f, %f, %f, %f];\n\n", q2[0], q2[1], q2[2], q2[3]);

      //printf("q1old = [%f, %f, %f, %f];\n\n", q1old[0], q1old[1], q1old[2], q1old[3]);
      //printf("q2old = [%f, %f, %f, %f];\n\n", q2old[0], q2old[1], q2old[2], q2old[3]);


      double mass = 0;
      for (int c = 0; c < k; c++) {  // clusters

	double p_pfh = lambda * exp(-lambda * (M[c] - *(F[j])).norm());
	double p_q1 = .5*(bingham_pdf(q1, &B_rot[2*c]) + bingham_pdf(q1, &B_rot[2*c+1]));
	double p_q2 = .5*(bingham_pdf(q2, &B_rot[2*c]) + bingham_pdf(q2, &B_rot[2*c+1]));
	double p_bingham = p_q1*p_q2;
	mass += p_pfh * p_bingham;

	//printf("%% cluster %d:\n", c);
	//printf("p_pfh = %f\n", p_pfh);
	//printf("p_q1 = .5*(%f + %f) = %f\n", bingham_pdf(q1, &B_rot[2*c]),
	//       bingham_pdf(q1, &B_rot[2*c+1]), p_q1);
	//printf("p_q2 = .5*(%f + %f) = %f\n", bingham_pdf(q2, &B_rot[2*c]),
	//       bingham_pdf(q2, &B_rot[2*c+1]), p_q2);
	//printf("p_bingham = %f\n\n", p_bingham);

      }

      //printf("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");

      //sleep(10);

      //printf("mass = %f\n", mass);
      pmf.mass[i] += log(mass);

      //break;  //dbug!!!
    }
    //printf("pmf.mass[i] = %f\n", pmf.mass[i]);
    //pmf.mass[i] = exp(pmf.mass[i]);
  }

  // normalize
  double logpmf_max = max(pmf.mass, pmf.n);
  for (int i = 0; i < pmf.n; i++)
    pmf.mass[i] = exp(pmf.mass[i] - logpmf_max);
  printf("sum(pmf.mass) = %f\n", sum(pmf.mass, pmf.n));
  mult(pmf.mass, pmf.mass, 1/sum(pmf.mass, pmf.n), pmf.n);

  printf("pmf.points = [ ...\n");
  for (int i = 0; i < pmf.n; i++)
    printf("%f %f %f %f ; ...\n", pmf.points[i][0], pmf.points[i][1], pmf.points[i][2], pmf.points[i][3]);
  printf("];\n\n");
  printf("pmf.mass = [ ");
  for (int i = 0; i < pmf.n; i++)
    printf("%f ...\n", pmf.mass[i]);
  printf("];\n\n");

  int imax = 0;
  double pmax = 0;
  for (int i = 0; i < pmf.n; i++) {
    if (pmf.mass[i] > pmax) {
      pmax = pmf.mass[i];
      imax = i;
    }
  }
  printf("q = [%f %f %f %f];\n\n", pmf.points[imax][0], pmf.points[imax][1], pmf.points[imax][2], pmf.points[imax][3]);
}


void estimatePose(PointCloud model_cloud, PointCloud observed_cloud)
{
  // initialize bingham library
  ros::Time ts = ros::Time::now();
  bingham_init();
  ROS_DEBUG("Initialized Bingham library in %f seconds.", (ros::Time::now() - ts).toSec());

  int n = model_cloud.get_points_size();

  int *L = getLabels(model_cloud);

  // get the number of clusters, k
  int k = 1;
  for (int i = 0; i < n; i++)
    if (L[i] + 1 > k)
      k = L[i] + 1;

  // get cluster means
  VectorXd *M = getClusterMeans(model_cloud, L, k);

  // get binghams
  bingham_t *B = getBinghams(model_cloud, L, k);

  // vote for pose
  poseVoting(observed_cloud, M, B, k);
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_model> <pcd_observed>", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 3)
    usage(argc, argv);

  char *fmodel = argv[1];
  char *fobs = argv[2];
  PointCloud model_cloud;
  PointCloud observed_cloud;
  cloud_io::loadPCDFile(fmodel, model_cloud);
  cloud_io::loadPCDFile(fobs, observed_cloud);

  ROS_DEBUG("Loaded PCD file '%s' with %d points and %d channels.",
	   fmodel, model_cloud.get_points_size(), model_cloud.get_channels_size());

  estimatePose(model_cloud, observed_cloud);

  return 0;
}
