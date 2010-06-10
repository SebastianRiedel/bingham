/*
 *  Common IO routines
 *  Copywrong (K) 2007 R.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: CommonIORoutines.h,v 1.1 2006/10/04 12:00:00 radu Exp $
 */
 
#ifndef INCLUDE_COMMONIOROUTINES_H
#define INCLUDE_COMMONIOROUTINES_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <string.h>
#include <algorithm>
#include <vector>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#define PCD_ASCII  0
#define PCD_BINARY 1

#include "ANN.h"
//typedef float  ANNcoord;
//typedef ANNcoord* ANNpoint;
//typedef ANNpoint* ANNpointArray;

struct PCD_Header
{
  // Column indices (eg. x=0, y=1, z=2, nx=3, ny=4, nz=5, c=6 for a XYZ-nXnYnZ-C file)
  std::vector<std::pair<int, std::string> > dimID;
  int nr_points;

  // Data type: PCD_ASCII = 0, PCD_BINARY = 1
  int data_type;

  // Defines the number of bytes in memory a point in the data file actually contains
  int point_size;
  //date
  //sensor
  //resolution (angular, distance)
  //max_range
  //pose_rel
  //pose_abs

  // Minimum and maximum values on each dimension
  ANNpoint minPD, maxPD;

  // Meta-comments
  std::string comments;
};

class DimIDComp
{
  std::string s;

  public:
    DimIDComp (std::string x) : s (x) { }
  
    bool operator () (std::pair<int, std::string> p)
    {
      return s == p.second;
    }
};


ANNpointArray         LoadPCDFile (const char* fileName, PCD_Header &header);
std::vector<ANNpoint> LoadPCDFileAsVector (const char* fileName, PCD_Header &header);

void readComments (std::string line, PCD_Header &header);

int           SavePCDFile (const char* fileName, ANNpointArray points, PCD_Header header, int precision);
int           SavePCDFile (const char* fileName, std::vector<ANNpoint> points, PCD_Header header, int precision);

int           SaveASCIIPCDFile  (const char* fileName, ANNpointArray points, PCD_Header header, int precision);
int           SaveASCIIPCDFile  (const char* fileName, ANNpointArray points, std::vector<int> indices, PCD_Header header, int precision);
int           SaveASCIIPCDFile  (const char* fileName, ANNpointArray points, std::set<int> indices, PCD_Header header, int precision);
int           SaveBinaryPCDFile (const char* fileName, ANNpointArray points, PCD_Header header);


std::string   createNewHeader ();
std::string   createNewHeaderLight ();
std::string   createNewFieldsHeader (PCD_Header header);
std::string   addCurrentHeader (PCD_Header header);
std::string   getAvailableDimensions (PCD_Header header);

// Check for duplicate columns
std::vector<int> checkForColumnDuplicates (PCD_Header header);

// Add a comment to the PCD header (maximum size: 256 characters)
void addCommentToHeader (PCD_Header &header, const char *format, ...);

int ParseDimensionArgument (int argc, char** argv, const char* str, std::string &val, PCD_Header header);

int getIndex (PCD_Header header, std::string value);

int getMaximumFeatureDimension (PCD_Header header, std::string value);

int  computeDimSize (std::string value);
void computePointSize (PCD_Header &header);
#endif
