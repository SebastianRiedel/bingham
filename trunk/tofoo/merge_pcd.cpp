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

@b Merge a set of PCD files.

 **/


// point cloud mapping
//#include <point_cloud_mapping/cloud_io.h>

#include "pointcloud.h"


#define ROS_INFO printf
#define ROS_ERROR printf


using namespace std;
//using namespace sensor_msgs;




/*
 * Note: mergePointClouds() assumes that all the clouds have the same channels.
 */
void mergePointClouds(PointCloud &dst, PointCloud *src, int n)
{
  // get the total number of points
  int pointsTotal = 0;
  for (int i = 0; i < n; i++)
    pointsTotal += src[i].get_points_size();

  //dst.header = src[0].header;
  dst.points.resize(pointsTotal);
  dst.channels.resize(src[0].channels.size());

  for (uint c = 0; c < dst.channels.size(); c++) {
    dst.channels[c].name = src[0].channels[c].name;
    dst.channels[c].values.resize(pointsTotal);
  }

  // Copy the data
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    for (uint j = 0; j < src[i].get_points_size(); j++) {
      dst.points[cnt].x = src[i].points[j].x;
      dst.points[cnt].y = src[i].points[j].y;
      dst.points[cnt].z = src[i].points[j].z;
      for (uint c = 0; c < dst.channels.size(); c++)
	dst.channels[c].values[cnt] = src[i].channels[c].values[j];
      cnt++;
    }
  }
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_merged> <pcd1> [<pcd2> ...]", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 3)
    usage(argc, argv);

  char *fout = argv[1];

  int n = argc - 2;
  char *fin[n];
  PointCloud clouds[n];
  for (int i = 0; i < n; i++) {
    fin[i] = argv[i+2];
    loadPCDFile(fin[i], clouds[i]);  //cloud_io::loadPCDFile(fin[i], clouds[i]);
    ROS_INFO("Loaded PCD file '%s' with %d points and %d channels.",
	     fin[i], clouds[i].get_points_size(), clouds[i].get_channels_size());
  }

  PointCloud mergedCloud;
  mergePointClouds(mergedCloud, clouds, n);

  ROS_INFO("Saving PCD file '%s' with %d points and %d channels.",
	   fout, mergedCloud.get_points_size(), mergedCloud.get_channels_size());

  int precision = 10;
  savePCDFileASCII(fout, mergedCloud, precision);  //cloud_io::savePCDFileASCII(fout, mergedCloud, precision);

  return 0;
}
