/*
 * Copyright (c) 2010 Jared Glover <jglov -=- mit.edu>
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
 */

/**
@mainpage

@htmlinclude manifest.html

\author Jared Glover

@b point cloud class

 **/


#include "pointcloud.h"
#include <assert.h>


/*
PointCloud::PointCloud(float **data, PCD_Header header)
{
  int npoints = header.nr_points;
  int nchannels = header.dimID.size();

  //std::vector<std::pair<int, std::string> > dimID;

  points.resize(npoints);
  channels.resize(nchannels);

  int ch_x = -1;
  int ch_y = -1;
  int ch_z = -1;

  for (int c = 0; c < nchannels; c++) {
    std::string name = header.dimID[c].second;
    channels[c].name = name;
    channels[c].values.resize(npoints);
    if (name.compare("x") == 0)
      ch_x = header.dimID[c].first;
    else if (name.compare("y") == 0)
      ch_y = header.dimID[c].first;
    else if (name.compare("z") == 0)
      ch_z = header.dimID[c].first;
  }

  if (ch_x < 0 || ch_y < 0 || ch_z < 0) {
    fprintf(stderr, "Error: Can't find x,y,z channels in PCD Header\n");
    exit(1);
  }

  for (int i = 0; i < npoints; i++) {
    points[i].x = data[i][ch_x];
    points[i].y = data[i][ch_y];
    points[i].z = data[i][ch_z];
    for (int c = 0; c < nchannels; c++)
      channels[c].values[i] = data[i][c];
  }
}
*/


int PointCloud::get_channels_size()
{
  return channels.size();
}

int PointCloud::get_points_size()
{
  return points.size();
}

void PointCloud::set_data_and_header(float **data, PCD_Header header)
{
  int npoints = header.nr_points;
  int nchannels = header.dimID.size();

  //std::vector<std::pair<int, std::string> > dimID;

  points.resize(npoints);
  channels.resize(nchannels);

  int ch_x = -1;
  int ch_y = -1;
  int ch_z = -1;

  for (int c = 0; c < nchannels; c++) {
    std::string name = header.dimID[c].second;
    channels[c].name = name;
    channels[c].values.resize(npoints);
    if (name.compare("x") == 0)
      ch_x = header.dimID[c].first;
    else if (name.compare("y") == 0)
      ch_y = header.dimID[c].first;
    else if (name.compare("z") == 0)
      ch_z = header.dimID[c].first;
  }

  if (ch_x < 0 || ch_y < 0 || ch_z < 0) {
    fprintf(stderr, "Error: Can't find x,y,z channels in PCD Header\n");
    exit(1);
  }

  for (int i = 0; i < npoints; i++) {
    points[i].x = data[i][ch_x];
    points[i].y = data[i][ch_y];
    points[i].z = data[i][ch_z];
    for (int c = 0; c < nchannels; c++)
      channels[c].values[i] = data[i][c];
  }
}

void PointCloud::get_header(PCD_Header &h)
{
  h.nr_points = points.size();

  h.dimID.resize(channels.size());
  for (int c = 0; c < channels.size(); c++) {
    h.dimID[c].first = c;
    h.dimID[c].second = channels[c].name;
  }
}

float **PointCloud::get_data()
{
  float **data = (float **)annAllocPts(points.size(), channels.size());
  assert(data != NULL);

  for (int i = 0; i < points.size(); i++)
    for (int c = 0; c < channels.size(); c++)
      data[i][c] = channels[c].values[i];

  return data;
}


void loadPCDFile(const char *fname, PointCloud &cloud)
{
  PCD_Header header;
  float **data = LoadPCDFile(fname, header);
  cloud.set_data_and_header(data, header);
  free(data);
}


void savePCDFileASCII(const char *fname, PointCloud cloud, int precision)
{
  PCD_Header header;
  cloud.get_header(header);
  float **data = cloud.get_data();
  SaveASCIIPCDFile(fname, data, header, precision);
  free(data);
}
