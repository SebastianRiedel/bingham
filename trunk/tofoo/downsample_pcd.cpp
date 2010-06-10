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

@b Downsample a PCD file.

 **/


// point cloud mapping
#include <point_cloud_mapping/cloud_io.h>
#include <point_cloud_mapping/geometry/point.h>


using namespace std;
using namespace sensor_msgs;


void downsample(PointCloud &dst, PointCloud &src, double sample_ratio)
{
  int num_points = (int)(src.get_points_size() * sample_ratio);

  if (num_points < 1)
    num_points = 1;

  int step = src.get_points_size() / num_points;

  dst.header = src.header;
  dst.points.resize(num_points);
  dst.channels.resize(src.channels.size());

  for (uint c = 0; c < dst.channels.size(); c++) {
    dst.channels[c].name = src.channels[c].name;
    dst.channels[c].values.resize(num_points);
  }

  // Copy the data
  int cnt = 0;
  for (uint i = step-1; i < src.get_points_size(); i += step) {
    dst.points[cnt].x = src.points[i].x;
    dst.points[cnt].y = src.points[i].y;
    dst.points[cnt].z = src.points[i].z;
    for (uint c = 0; c < dst.channels.size(); c++)
      dst.channels[c].values[cnt] = src.channels[c].values[i];
    cnt++;
  }
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <pcd_in> <pcd_out> <sample_ratio>", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 4)
    usage(argc, argv);

  char *fin = argv[1];
  char *fout = argv[2];
  double sample_ratio = atof(argv[3]);
  
  if (sample_ratio < 0 || sample_ratio > 1)
    usage(argc, argv);

  PointCloud cloud, cloud_down;
  loadPCDFile(fin, cloud);  //cloud_io::loadPCDFile(fin, cloud);

  ROS_INFO("Loaded PCD file '%s' with %d points and %d channels.",
	   fin, cloud.get_points_size(), cloud.get_channels_size());

  downsample(cloud_down, cloud, sample_ratio);

  int precision = 10;
  savePCDFileASCII(fout, cloud_down, precision);  //cloud_io::savePCDFileASCII(fout, cloud_down, precision);

  return 0;
}
