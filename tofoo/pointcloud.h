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


#ifndef INCLUDE_POINTCLOUD_H
#define INCLUDE_POINTCLOUD_H


#include <vector>
#include <string>
#include "CommonIORoutines.h"


typedef struct {
  float x;
  float y;
  float z;
} Point;

typedef struct {
  std::string name;
  std::vector<float> values;
} Channel;



class PointCloud
{
 public:

  //PointCloud(float **data, PCD_Header header);


  std::vector<Point> points;
  std::vector<Channel> channels;

  int get_channels_size();
  int get_points_size();
  float **get_data();
  void get_header(PCD_Header &h);
  void set_data_and_header(float **data, PCD_Header header);
};


void loadPCDFile(const char *fname, PointCloud &cloud);
void savePCDFileASCII(const char *fname, PointCloud cloud, int precision);



#endif  //INCLUDE_POINTCLOUD_H

