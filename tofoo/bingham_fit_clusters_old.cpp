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

@b Fit bingham distributions to the local coordinate frames of each cluster.

 **/



// eigen
#include <Eigen/Core>
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
static void randomBingham(bingham_t *B, double z0)
{
  double Z[3] = {z0, z0, z0};
  double V[3][4] = {{0,0,0,0}, {0,0,0,0}, {0,0,0,0}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};
  for (int i = 0; i < 4; i++) {
    V[0][i] = .99*frand() + .01;
    if (frand() > .5)
      V[0][i] = -V[0][i];
  }
  double d = sqrt(V[0][0]*V[0][0] + V[0][1]*V[0][1] + V[0][2]*V[0][2] + V[0][3]*V[0][3]);
  for (int i = 0; i < 4; i++)
    V[0][i] /= d;
  V[1][0] = V[0][1];
  V[1][1] = -V[0][0];
  V[2][2] = V[0][3];
  V[2][3] = -V[0][2];

  bingham_new(B, 4, Vp, Z);
}
*/


void fitBimodalBingham(bingham_t *B1, bingham_t *B2, Vector4d *Q, int *L, int n, int c)
{
  // get num points in cluster c
  int cnt = 0;
  for (int i = 0; i < n; i++)
    if (L[i] == c)
      cnt++;

  if (cnt < 3) {
    ROS_ERROR("Cluster %d has %d points, but at least 3 points are needed to fit a Bingham", c, cnt);
    return;
  }

  // there are 2 quaternions for each point
  cnt *= 2;

  // get all quaternions in Q belonging to class c
  double *X[cnt];
  double *X_raw = (double *)calloc(cnt*4, sizeof(double));
  for (int i = 0; i < cnt; i++)
    X[i] = X_raw + 4*i;
  cnt = 0;
  for (int i = 0; i < n; i++) {
    if (L[i] == c) {
      X[cnt][0] = Q[2*i][0];
      X[cnt][1] = Q[2*i][1];
      X[cnt][2] = Q[2*i][2];
      X[cnt][3] = Q[2*i][3];
      cnt++;
      X[cnt][0] = Q[2*i+1][0];
      X[cnt][1] = Q[2*i+1][1];
      X[cnt][2] = Q[2*i+1][2];
      X[cnt][3] = Q[2*i+1][3];
      cnt++;
    }
  }

  // run kmeans to fit a bimodal bingham distribution

  double z0 = -5;
  double Z[3] = {z0, z0, z0};
  double V[4][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp1[3] = {&V[0][0], &V[1][0], &V[2][0]};
  double *Vp2[3] = {&V[1][0], &V[2][0], &V[3][0]};
  bingham_new(B1, 4, Vp1, Z);
  bingham_new(B2, 4, Vp2, Z);

  //randomBingham(B1, -5);  // init cluster 1
  //randomBingham(B2, -5);  // init cluster 2

  double *X1[cnt];
  double *X2[cnt];

  int BL[cnt];  // bingham labels (0 or 1)

  double logf = DBL_MIN;  // log likelihood
  double logf_thresh = 1e-6;

  const int iter = 100;
  for (int j = 0; j < iter; j++) {
    // compute labels
    //printf("B = [ ");
    for (int i = 0; i < cnt; i++) {
      double p1 = bingham_pdf(X[i], B1);
      double p2 = bingham_pdf(X[i], B2);
      BL[i] = (p2 > p1);
      //printf("%d ", BL[i]);
    }
    //printf("]\n");

    // fit binghams
    int n1 = 0, n2 = 0;
    for (int i = 0; i < cnt; i++) {
      if (BL[i] == 0)
	X1[n1++] = X_raw + 4*i;
      else
	X2[n2++] = X_raw + 4*i;
    }
    bingham_fit(B1, X1, n1, 4);
    bingham_fit(B2, X2, n2, 4);

    // check if total log likelihood is converging
    double logf_prev = logf;
    logf = bingham_L(B1, X1, n1) + bingham_L(B2, X2, n2);

    //printf("logf = %f\n", logf);

    //ROS_INFO("B1(%f, %f, %f), B2(%f, %f, %f)",
    //	     B1->Z[0], B1->Z[1], B1->Z[2], B2->Z[0], B2->Z[1], B2->Z[2]);

    if (fabs(logf - logf_prev) < logf_thresh)
      break;
  }

  ROS_INFO("Fit Binghams B1(%f, %f, %f), B2(%f, %f, %f) to cluster %d with %d points.",
	   B1->Z[0], B1->Z[1], B1->Z[2], B2->Z[0], B2->Z[1], B2->Z[2], c, cnt);
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


int *getLabels(PointCloud cloud)
{
  int c = getChannel(cloud, "cluster");

  int n = cloud.get_points_size();
  int *L = (int *)malloc(n*sizeof(int));

  for (int i = 0; i < n; i++)
    L[i] = cloud.channels[c].values[i];

  return L;
}


void addBinghamChannels(PointCloud &cloud, bingham_t *B, int *L)
{
  int channels_per_bingham = 16;
  int c0 = cloud.get_channels_size();

  cloud.channels.resize(c0 + 2*channels_per_bingham);

  int c = c0;
  cloud.channels[c++].name = "b1f";
  cloud.channels[c++].name = "b1z1";
  cloud.channels[c++].name = "b1z2";
  cloud.channels[c++].name = "b1z3";
  cloud.channels[c++].name = "b1v11";
  cloud.channels[c++].name = "b1v12";
  cloud.channels[c++].name = "b1v13";
  cloud.channels[c++].name = "b1v14";
  cloud.channels[c++].name = "b1v21";
  cloud.channels[c++].name = "b1v22";
  cloud.channels[c++].name = "b1v23";
  cloud.channels[c++].name = "b1v24";
  cloud.channels[c++].name = "b1v31";
  cloud.channels[c++].name = "b1v32";
  cloud.channels[c++].name = "b1v33";
  cloud.channels[c++].name = "b1v34";

  cloud.channels[c++].name = "b2f";
  cloud.channels[c++].name = "b2z1";
  cloud.channels[c++].name = "b2z2";
  cloud.channels[c++].name = "b2z3";
  cloud.channels[c++].name = "b2v11";
  cloud.channels[c++].name = "b2v12";
  cloud.channels[c++].name = "b2v13";
  cloud.channels[c++].name = "b2v14";
  cloud.channels[c++].name = "b2v21";
  cloud.channels[c++].name = "b2v22";
  cloud.channels[c++].name = "b2v23";
  cloud.channels[c++].name = "b2v24";
  cloud.channels[c++].name = "b2v31";
  cloud.channels[c++].name = "b2v32";
  cloud.channels[c++].name = "b2v33";
  cloud.channels[c++].name = "b2v34";

  int n = cloud.get_points_size();
  for (int i = 0; i < 2*channels_per_bingham; i++)
    cloud.channels[c0+i].values.resize(n);

  for (int i = 0; i < n; i++) {
    c = c0;

    int j = L[i];

    cloud.channels[c++].values[i] = B[2*j].F;
    cloud.channels[c++].values[i] = B[2*j].Z[0];
    cloud.channels[c++].values[i] = B[2*j].Z[1];
    cloud.channels[c++].values[i] = B[2*j].Z[2];
    cloud.channels[c++].values[i] = B[2*j].V[0][0];
    cloud.channels[c++].values[i] = B[2*j].V[0][1];
    cloud.channels[c++].values[i] = B[2*j].V[0][2];
    cloud.channels[c++].values[i] = B[2*j].V[0][3];
    cloud.channels[c++].values[i] = B[2*j].V[1][0];
    cloud.channels[c++].values[i] = B[2*j].V[1][1];
    cloud.channels[c++].values[i] = B[2*j].V[1][2];
    cloud.channels[c++].values[i] = B[2*j].V[1][3];
    cloud.channels[c++].values[i] = B[2*j].V[2][0];
    cloud.channels[c++].values[i] = B[2*j].V[2][1];
    cloud.channels[c++].values[i] = B[2*j].V[2][2];
    cloud.channels[c++].values[i] = B[2*j].V[2][3];

    cloud.channels[c++].values[i] = B[2*j+1].F;
    cloud.channels[c++].values[i] = B[2*j+1].Z[0];
    cloud.channels[c++].values[i] = B[2*j+1].Z[1];
    cloud.channels[c++].values[i] = B[2*j+1].Z[2];
    cloud.channels[c++].values[i] = B[2*j+1].V[0][0];
    cloud.channels[c++].values[i] = B[2*j+1].V[0][1];
    cloud.channels[c++].values[i] = B[2*j+1].V[0][2];
    cloud.channels[c++].values[i] = B[2*j+1].V[0][3];
    cloud.channels[c++].values[i] = B[2*j+1].V[1][0];
    cloud.channels[c++].values[i] = B[2*j+1].V[1][1];
    cloud.channels[c++].values[i] = B[2*j+1].V[1][2];
    cloud.channels[c++].values[i] = B[2*j+1].V[1][3];
    cloud.channels[c++].values[i] = B[2*j+1].V[2][0];
    cloud.channels[c++].values[i] = B[2*j+1].V[2][1];
    cloud.channels[c++].values[i] = B[2*j+1].V[2][2];
    cloud.channels[c++].values[i] = B[2*j+1].V[2][3];
  }

  ROS_INFO("Added bingham channels to point cloud\n");
}


void fitBinghams(PointCloud &cloud)
{
  // initialize bingham library
  ros::Time ts = ros::Time::now();
  bingham_init();
  ROS_DEBUG("Initialized Bingham library in %f seconds.", (ros::Time::now() - ts).toSec());

  int n = cloud.get_points_size();

  int ch_nx = getChannel(cloud, "nx");
  int ch_pcx = getChannel(cloud, "pcx");
  int ch_cluster = getChannel(cloud, "cluster");
  
  if (ch_nx < 0 || ch_pcx < 0 || ch_cluster < 0) {
    ROS_ERROR("The PCD file must contain all of the following channels: "
	      "nx, ny, nz, pcx, pcy, pcz, cluster.\n");
    exit(1);
  }

  int *L = getLabels(cloud);
  Vector4d *Q = getQuaternions(cloud);

  // get the number of clusters, k
  int k = 1;
  for (int i = 0; i < n; i++)
    if (L[i] + 1 > k)
      k = L[i] + 1;

  // fit binghams
  bingham_t B[2*k];
  for (int i = 0; i < k; i++)
    fitBimodalBingham(&B[2*i], &B[2*i+1], Q, L, n, i);

  // add bingham channels
  addBinghamChannels(cloud, B, L);
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_in> <pcd_out>", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 3)
    usage(argc, argv);

  char *fin = argv[1];
  char *fout = argv[2];
  PointCloud cloud;
  cloud_io::loadPCDFile(fin, cloud);

  ROS_DEBUG("Loaded PCD file '%s' with %d points and %d channels.",
	   fin, cloud.get_points_size(), cloud.get_channels_size());

  //printf("\n\n");
  //ROS_DEBUG("channels:");
  //for (uint i = 0; i < cloud.get_channels_size(); i++) {
  //  ROS_DEBUG("  - %s", cloud.channels[i].name.c_str());
  //}
  //printf("\n\n");

  fitBinghams(cloud);

  int precision = 10;
  cloud_io::savePCDFileASCII(fout, cloud, precision);

  return 0;
}
