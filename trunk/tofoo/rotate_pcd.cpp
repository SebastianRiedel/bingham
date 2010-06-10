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

@b Rotate a PCD file.

 **/


// eigen
#include <Eigen/Core>
#include <Eigen/LU>
USING_PART_OF_NAMESPACE_EIGEN
//#define EIGEN_USE_NEW_STDVECTOR
//#include <Eigen/StdVector>


// point cloud mapping
#include <point_cloud_mapping/cloud_io.h>


using namespace std;
using namespace sensor_msgs;


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


int getChannel(PointCloud cloud, const char *name)
{
  for (uint i = 0; i < cloud.get_channels_size(); i++)
    if (!cloud.channels[i].name.compare(name))
      return i;

  return -1;
}


void rotatePoint(float &x, float &y, float &z, Matrix3d &R)
{
  Vector3d v(x,y,z);

  v = R*v;

  x = v[0];
  y = v[1];
  z = v[2];
}


void rotate(PointCloud &cloud, Vector4d q)
{
  int n = cloud.get_points_size();

  Matrix3d R = quaternionToRotationMatrix(q);

  int ch_nx = getChannel(cloud, "nx");
  int ch_ny = getChannel(cloud, "ny");
  int ch_nz = getChannel(cloud, "nz");
  int ch_pcx = getChannel(cloud, "pcx");
  int ch_pcy = getChannel(cloud, "pcy");
  int ch_pcz = getChannel(cloud, "pcz");

  // rotate the points
  for (int i = 0; i < n; i++)
    rotatePoint(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z, R);

  // rotate the normals
  if (ch_nx >= 0)
    for (int i = 0; i < n; i++)
      rotatePoint(cloud.channels[ch_nx].values[i],
		  cloud.channels[ch_ny].values[i],
		  cloud.channels[ch_nz].values[i], R);

  // rotate the principle curvatures
  if (ch_pcx >= 0)
    for (int i = 0; i < n; i++)
      rotatePoint(cloud.channels[ch_pcx].values[i],
		  cloud.channels[ch_pcy].values[i],
		  cloud.channels[ch_pcz].values[i], R);
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_in> <pcd_out> <qw> <qx> <qy> <qz>", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 7)
    usage(argc, argv);

  char *fin = argv[1];
  char *fout = argv[2];

  Vector4d q;
  q[0] = atof(argv[3]);
  q[1] = atof(argv[4]);
  q[2] = atof(argv[5]);
  q[3] = atof(argv[6]);
  q.normalize();
  
  PointCloud cloud;
  cloud_io::loadPCDFile(fin, cloud);

  ROS_INFO("Loaded PCD file '%s' with %d points and %d channels.",
	   fin, cloud.get_points_size(), cloud.get_channels_size());

  rotate(cloud, q);

  int precision = 10;
  cloud_io::savePCDFileASCII(fout, cloud, precision);

  return 0;
}
