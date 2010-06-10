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

@b K-Means clustering on the PFHs in a PCD file.

 **/



#include <float.h>

// eigen
#include "eigen/Eigen/Core"
USING_PART_OF_NAMESPACE_EIGEN
//#define EIGEN_USE_NEW_STDVECTOR
//#include <Eigen/StdVector>



// point cloud mapping
//#include <point_cloud_mapping/cloud_io.h>

#include "pointcloud.h"


#define ROS_INFO printf
#define ROS_ERROR printf
#define ROS_DEBUG printf




using namespace std;
//using namespace sensor_msgs;


int getChannel(PointCloud &cloud, const char *name)
{
  for (uint i = 0; i < cloud.get_channels_size(); i++)
    if (!cloud.channels[i].name.compare(name))
      return i;

  return -1;
}


VectorXd *getPFH(PointCloud &cloud, int index)
{
  // find the channel of the first feature coordinate
  int c = getChannel(cloud, "f1");
  int n = cloud.get_channels_size() - c;

  VectorXd *pfh = new VectorXd(n);

  for (int i = 0; i < n; i++)
    (*pfh)[i] = cloud.channels[c+i].values[index];

  return pfh;
}


// returns SSD
double kmeans(int *L, VectorXd **X, int n, int k)
{
  const double ssd_thresh = 1e-4;
  const int iter = 1000;

  const int p = X[0]->size();  // all vectors have the same length
  const VectorXd zeros = VectorXd::Zero(p);

  // initialize centroids M by randomly sampling from X
  VectorXd *M[k];
  for (int i = 0; i < k; i++) {
    int r = rand() % n;
    M[i] = new VectorXd(p);
    *(M[i]) = *(X[r]);
  }

  double ssd = DBL_MAX;
  
  for (int cnt = 0; cnt < iter; cnt++) {

    double ssd_prev = ssd;

    // update L
    for (int i = 0; i < n; i++) {
      double dmin = DBL_MAX;
      int cmin = 0;
      for (int c = 0; c < k; c++) {
	VectorXd dx = *(X[i]) - *(M[c]);
	double d = dx.dot(dx);
	if (d < dmin) {
	  dmin = d;
	  cmin = c;
	}
      }
      L[i] = cmin;
    }

    // update M
    for (int c = 0; c < k; c++) {
      *(M[c]) = zeros;
      int npoints = 0;
      for (int i = 0; i < n; i++) {
	if (L[i] == c) {
	  *(M[c]) += *(X[i]);
	  npoints++;
	}
      }
      *(M[c]) /= (double)npoints;
    }
    
    // compute sum of squared distances
    ssd = 0;
    for (int c = 0; c < k; c++) {
      for (int i = 0; i < n; i++) {
	if (L[i] == c) {
	  VectorXd dx = *(X[i]) - *(M[c]);
	  ssd += dx.dot(dx);
	}
      }
    }

    if (fabs(ssd - ssd_prev) < ssd_thresh)
      break;
  }

  for (int i = 0; i < k; i++)
    delete M[i];

  return ssd;
}


// returns num. clusters
int kmeansBIC(int *L_min, VectorXd **X, int n)
{
  const int num_restarts = 5;
  const int max_clusters = 20;

  int p = X[0]->size();  // all vectors have the same length

  double bic_min = DBL_MAX;
  int kmin = 0;
  int L[n];

  for (int k = 2; k < max_clusters; k++) {

    // if we haven't improved in the last 5 iterations, break
    if (k - kmin >= 5)
      break;

    printf("Trying k = %d...", k);

    double ssd, ssd_min = DBL_MAX;
    double imin = 0;
    int Lk_min[n];
    for (int i = 0; i < num_restarts; i++) {
      ssd = kmeans(L, X, n, k);
      if (ssd < ssd_min) {
	ssd_min = ssd;
	imin = i;
	memcpy(Lk_min, L, n*sizeof(int));
      }
    }
    ssd = ssd_min;
    memcpy(L, Lk_min, n*sizeof(int));

    double bic = n*(1 + log(ssd)) + (k*p-n)*log(n);
    
    printf("bic = %f", bic);

    if (bic < bic_min) {
      printf(" *** new best ***");
      bic_min = bic;
      kmin = k;
      memcpy(L_min, L, n*sizeof(int));
    }
    printf("\n");
  }

  return kmin;
}


int cluster(int *L, VectorXd **X, int n, int k)
{
  if (k < 0)
    return kmeansBIC(L, X, n);

  const int num_restarts = 5;
  double ssd, ssd_min = DBL_MAX;
  double imin = 0;
  int Lk_min[n];
  for (int i = 0; i < num_restarts; i++) {
    ssd = kmeans(L, X, n, k);
    if (ssd < ssd_min) {
      ssd_min = ssd;
      imin = i;
      memcpy(Lk_min, L, n*sizeof(int));
    }
  }
  ssd = ssd_min;
  memcpy(L, Lk_min, n*sizeof(int));

  return k;
}


void clusterPFH(PointCloud &cloud, int k)
{
  // get an array of PFH vectors (for each point)
  int n = cloud.get_points_size();
  VectorXd *F[n];
  for (int i = 0; i < n; i++)
    F[i] = getPFH(cloud, i);

  // cluster PFH vectors with KMeans
  int labels[n];
  k = cluster(labels, F, n, k);

  ROS_INFO("Clustered PFHs into %d clusters", k);

  int c = getChannel(cloud, "cluster");
  if (c < 0) {
    c = cloud.get_channels_size();
    cloud.channels.resize(c+1);
    cloud.channels[c].name = "cluster";
    cloud.channels[c].values.resize(n);
  }
  for (int i = 0; i < n; i++)
    cloud.channels[c].values[i] = labels[i];

  ROS_INFO("Added channel 'cluster' to point cloud\n");
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_in> <pcd_out> [num_clusters]", argv[0]);
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

  int k = -1;
  if (argc >= 4)
    k = atoi(argv[3]);

  // dbug
  //printf("\n\n");
  //printf("channels:");
  //for (uint i = 0; i < cloud.get_channels_size(); i++) {
  //  printf("  - %s", cloud.channels[i].name.c_str());
  //}
  //printf("\n\n");

  clusterPFH(cloud, k);

  int precision = 10;
  savePCDFileASCII(fout, cloud, precision);  //cloud_io::savePCDFileASCII(fout, cloud, precision);

  return 0;
}
