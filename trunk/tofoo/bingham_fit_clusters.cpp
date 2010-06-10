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
#include "eigen/Eigen/Core"
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

#include "pointcloud.h"

#define ROS_INFO printf
#define ROS_ERROR printf
#define ROS_DEBUG printf


using namespace std;
//using namespace sensor_msgs;


int getChannel(PointCloud cloud, const char *name)
{
  for (uint i = 0; i < cloud.get_channels_size(); i++)
    if (!cloud.channels[i].name.compare(name))
      return i;

  return -1;
}


void fitBinghamMixture(bingham_mix_t *BM, Vector4d *Q, int *L, int n, int c)
{
  // get num points in cluster c
  int cnt = 0;
  for (int i = 0; i < n; i++)
    if (L[i] == c)
      cnt++;

  if (cnt < 4) {
    ROS_ERROR("Cluster %d has %d points, but at least 4 points are needed to fit a Bingham", c, cnt);
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

  // fit binghams
  bingham_cluster(BM, X, cnt, 4);

  ROS_INFO("Fit %d Binghams to cluster %d with %d points.\n", BM->n, c, cnt);
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
    L[i] = (int)cloud.channels[c].values[i];

  return L;
}


/*
 * Write bingham mixtures in the following format:
 *
 * B <c> <i> <w> <d> <F> <Z> <V>
 *
void writeBinghamMixtures(bingham_mix_t *BM, int num_clusters, char *fout)
{
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

void fitBinghams(PointCloud &cloud, char *fout)
{
  // initialize bingham library
  //ros::Time ts = ros::Time::now();
  bingham_init();
  //ROS_DEBUG("Initialized Bingham library in %f seconds.", (ros::Time::now() - ts).toSec());

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

  // fit bingham mixtures
  bingham_mix_t BM[k];
  for (int i = 0; i < k; i++)
    fitBinghamMixture(&BM[i], Q, L, n, i);

  // write bingham mixtures to file
  //writeBinghamMixtures(BM, k, fout);
  save_bmx(BM, k, fout);
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_in> <bmx_out>", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 3)
    usage(argc, argv);

  char *fin = argv[1];
  char *fout = argv[2];
  PointCloud cloud;
  loadPCDFile(fin, cloud);  //cloud_io::loadPCDFile(fin, cloud);

  ROS_DEBUG("Loaded PCD file '%s' with %d points and %d channels.",
	   fin, cloud.get_points_size(), cloud.get_channels_size());

  fitBinghams(cloud, fout);

  return 0;
}
