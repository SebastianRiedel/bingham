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
 * $Id: CommonIORoutines.cc,v 1.1 2006/10/04 12:00:00 radu Exp $
 */

/// A few warnings regarding the new binary PCD v2 .pcd format:
///
/// @1: -> this will not be cross-platform, in the sense that it will only work right now if 
///     sizeof(int) sizeof(float), sizeof (unsigned char), etc are the same on different platforms
///
/// @2: -> we are using the fastest possible way of reading/writing data to files using mmap/munmap. If this is not fast 
///     enough, then we should look at creating virtual RAM disks and storing data sets there for the duration of a trial
///
/// @3: -> remember that we ASSUME that ANNpoint is now a pointer to a float and not a double!
///
/// @4: -> after playing around with memcpy's, I decided to sacrifice some size (meaning that all values, even RGB colors
///     will be saved as floats even though they could be saved as unsigned chars) in order to get the fastest saving 
///     and loading time.

#include "CommonIORoutines.h"
#include "StringTokenizer.h"
#include <stdarg.h>

////////////////////////////////////////////////////////////////////////////////
// Load point cloud data from a PCD file containing n-D points
// Determine automatically if file contains ASCII or BINARY data
/// @warning: header size limited to maximum getpagesize (4096 on most systems)
/// @note: all lines besides:
// - the ones beginning with # (treated as comments
// - FIELDS/COLUMNS ...
// - POINTS ...
// - DATA ...
// ...are intepreted as data points. Failure to comply with this simple rule might result in errors and warnings! :)
ANNpointArray
  LoadPCDFile (const char* fileName, PCD_Header &header)
{
  int data_idx = 0;
  header.data_type = PCD_ASCII;
  header.nr_points = 0;
  header.dimID.clear ();
  header.minPD = header.maxPD = NULL;
  
  ANNpointArray points = NULL;
  std::ifstream fs;
  std::string line;

  int idx = 0, nrld = 0;
  // Open file
  fs.open (fileName);
  if (!fs.is_open () || fs.fail ())
  {
    fprintf (stderr, "Couldn't open %s for reading!\n", fileName);
    return points;
  }
  // Read the header and fill it in with wonderful values
  while (!fs.eof ())
  {
    getline (fs, line);
    if (line == "")
      continue;

    StringTokenizer st = StringTokenizer (line, " ");

    std::string lineType = st.nextToken ();

    /// ---[ Perform checks to see what does this line represents
    if (lineType.substr (0, 1) == "#")
    {
//      readComments (line, header);
      continue;
    }
    // Get the column indices
    if ((lineType.substr (0, 7) == "COLUMNS") || (lineType.substr (0, 6) == "FIELDS"))
    {
      int remainingTokens = st.countTokens ();
      for (int i = 0; i < remainingTokens; i++)
      {
        std::string colType = st.nextToken ();
        std::pair <int, std::string> column;
        column.first  = i;
        column.second = colType;
        header.dimID.push_back (column);
      }
      continue;
    }

    // Ignore COUNT, SIZE, TYPE, WIDTH and HEIGHT for now
    if (lineType.substr (0, 5) == "COUNT") { readComments (line, header); continue; }
    if (lineType.substr (0, 4) == "SIZE") { readComments (line, header); continue; }
    if (lineType.substr (0, 4) == "TYPE") { readComments (line, header); continue; }
    if (lineType.substr (0, 5) == "WIDTH") { readComments (line, header); continue; }
    if (lineType.substr (0, 6) == "HEIGHT") { readComments (line, header); continue; }

    // Get the number of points
    if (lineType.substr (0, 6) == "POINTS")
    {
      header.nr_points = st.nextIntToken ();
      continue;
    }

    // Allocate memory for all the points
    if ((header.nr_points != 0) && (header.dimID.size () != 0) && (points == NULL))
    {
      points = annAllocPts (header.nr_points, header.dimID.size ());
      header.minPD = annAllocPt (header.dimID.size ());
      header.maxPD = annAllocPt (header.dimID.size ());
      for (unsigned int i = 0; i < header.dimID.size (); i++)
      {
        header.minPD[i] = FLT_MAX;
        header.maxPD[i] = -FLT_MAX;
      }
    }

    // Check DATA type
    if (lineType.substr (0, 4) == "DATA")
    {
      std::string dataType = st.nextToken ();
      if (dataType.substr (0, 6) == "binary")
      {
        fprintf (stderr, "(binary)");
        header.data_type = PCD_BINARY;
        data_idx = fs.tellg ();
#ifdef ANN_DOUBLE
        fprintf (stderr, "Warning! ANN_DOUBLE defined while reading a binary file! Are you sure you know what you're doing?\n");
#endif
        break;
      }
      fprintf (stderr, "(ascii)");
      continue;
    }

    // Parse for additional header information

    /// ---[ ASCII mode only
    // Nothing of the above? We must have points then
    /// ---[ Using allocated points like this is bad because we cannot deallocate them
    ///ANNpoint p = annAllocPt (header.dimID.size ());
    // Convert the first token to float and use it as the first point coordinate
    if (idx >= header.nr_points)
    {
      fprintf (stderr, "Error: input file %s has more points than advertised (%d)!\n", fileName, header.nr_points);
      break;
    }
    points[idx][0] = atof (lineType.c_str ());
    header.minPD[0] = (points[idx][0] < header.minPD[0]) ? points[idx][0] : header.minPD[0];
    header.maxPD[0] = (points[idx][0] > header.maxPD[0]) ? points[idx][0] : header.maxPD[0];
    for (unsigned int i = 1; i < header.dimID.size (); i++)
    {
      points[idx][i] = st.nextFloatToken ();
      header.minPD[i] = (points[idx][i] < header.minPD[i]) ? points[idx][i] : header.minPD[i];
      header.maxPD[i] = (points[idx][i] > header.maxPD[i]) ? points[idx][i] : header.maxPD[i];
    }
    idx++;
    /// ---]
  }
  // Close file
  fs.close ();

  // Compute how much a point (with it's N-dimensions) occupies in terms of bytes
  computePointSize (header);

  /// ---[ Binary mode only
  /// We must re-open the file and read with mmap () for binary
  if (header.data_type == PCD_BINARY)
  {
    // Open for reading
    int fd = open (fileName, O_RDONLY);
    if (fd == -1)
      return (NULL);

    // Compute how much a point (with it's N-dimensions) occupies in terms of bytes
    //computePointSize (header);
    header.point_size = sizeof (float) * header.dimID.size ();

    // Prepare the map
    char *map = (char*)mmap (0, header.nr_points * header.point_size, PROT_READ, MAP_SHARED, fd, getpagesize ());
    if (map == MAP_FAILED)
    {
      close (fd);
      return (NULL);
    }

    // Prepare the dimensions sizes
    std::vector<int> dimsizes (header.dimID.size ());
    for (unsigned int j = 0; j < header.dimID.size (); j++)
      //dimsizes[j] = computeDimSize (header.dimID[j].second);
      dimsizes[j] = sizeof (float);

    // Read the data
    for (int i = 0; i < header.nr_points; i++)
    {
      int idx_j = 0;
      for (unsigned int j = 0; j < header.dimID.size (); j++)
      {
        memcpy (reinterpret_cast<ANNcoord*>(&points[i][j]), (char*)&map[i*header.point_size + idx_j], dimsizes[j]);
        idx_j += dimsizes[j];
        header.minPD[j] = (points[i][j] < header.minPD[j]) ? points[i][j] : header.minPD[j];
        header.maxPD[j] = (points[i][j] > header.maxPD[j]) ? points[i][j] : header.maxPD[j];
      }
    }

    // Unmap the pages of memory
    if (munmap (map, header.nr_points * header.point_size) == -1)
    {
      close (fd);
      return (NULL);
    }
    close (fd);
  }

  if ((idx != header.nr_points) && (header.data_type == PCD_ASCII))
  {
    fprintf (stderr, "\nWarning! Number of points read (");
    fprintf (stderr, "%d", idx); fprintf (stderr, ") is different than expected (");
    fprintf (stderr, "%d", header.nr_points); fprintf (stderr, ")!\n");
    header.nr_points = idx;
  }

  // Check for duplicate dimensions
  std::vector<int> c = checkForColumnDuplicates (header);
  if (c.size () != 0)
  {
    fprintf (stderr, "\nError: the following columns have a duplicate entry: ");
    for (unsigned int d = 0; d < c.size (); d++)
      fprintf (stderr, "%d(%s) ", header.dimID.at (c.at (d)).first, header.dimID.at (c.at (d)).second.c_str ());
    fprintf (stderr, "\n");
  }
  return points;
}
std::vector<ANNpoint>
  LoadPCDFileAsVector (const char* fileName, PCD_Header &header)
{
  int data_idx = 0;
  header.data_type = PCD_ASCII;
  header.nr_points = 0;
  header.dimID.clear ();
  header.minPD = header.maxPD = NULL;
  
  std::vector<ANNpoint> points;
  std::ifstream fs;
  std::string line;

  int idx = 0, nrld = 0;
  // Open file
  fs.open (fileName);
  if (!fs.is_open () || fs.fail ())
  {
    fprintf (stderr, "Couldn't open %s for reading!\n", fileName);
    return points;
  }
  // Read the header and fill it in with wonderful values
  while (!fs.eof ())
  {
    getline (fs, line);
    if (line == "")
      continue;

    StringTokenizer st = StringTokenizer (line, " ");

    std::string lineType = st.nextToken ();

    /// ---[ Perform checks to see what does this line represents
    if (lineType.substr (0, 1) == "#")
    {
      readComments (line, header);
      continue;
    }
    // Get the column indices
    if ((lineType.substr (0, 7) == "COLUMNS") || (lineType.substr (0, 6) == "FIELDS"))
    {
      int remainingTokens = st.countTokens ();
      for (int i = 0; i < remainingTokens; i++)
      {
        std::string colType = st.nextToken ();
        std::pair <int, std::string> column;
        column.first  = i;
        column.second = colType;
        header.dimID.push_back (column);
      }
      continue;
    }
    // Get the number of points
    if (lineType.substr (0, 6) == "POINTS")
    {
      header.nr_points = st.nextIntToken ();
      continue;
    }

    // Allocate memory for all the points
    if ((header.nr_points != 0) && (header.dimID.size () != 0) && (points.size () == 0))
    {
      points.resize (header.nr_points);
      header.minPD = annAllocPt (header.dimID.size ());
      header.maxPD = annAllocPt (header.dimID.size ());
      for (unsigned int i = 0; i < header.dimID.size (); i++)
      {
        header.minPD[i] = FLT_MAX;
        header.maxPD[i] = -FLT_MAX;
      }
    }

    // Check DATA type
    if (lineType.substr (0, 4) == "DATA")
    {
      std::string dataType = st.nextToken ();
      if (dataType.substr (0, 6) == "binary")
      {
        fprintf (stderr, "(binary)");
        header.data_type = PCD_BINARY;
        data_idx = fs.tellg ();
        break;
      }
      fprintf (stderr, "(ascii)");
      continue;
    }

    // Parse for additional header information

    /// ---[ ASCII mode only
    // Nothing of the above? We must have points then
    /// ---[ Using allocated points like this is bad because we cannot deallocate them
    ///ANNpoint p = annAllocPt (header.dimID.size ());
    // Convert the first token to float and use it as the first point coordinate
    if (idx >= header.nr_points)
    {
      fprintf (stderr, "Error: input file %s has more points than advertised (%d)!\n", fileName, header.nr_points);
      break;
    }
    ANNpoint pt = annAllocPt (header.dimID.size ());
    pt[0] = atof (lineType.c_str ());
    points[idx] = pt;
    header.minPD[0] = (pt[0] < header.minPD[0]) ? pt[0] : header.minPD[0];
    header.maxPD[0] = (pt[0] > header.maxPD[0]) ? pt[0] : header.maxPD[0];
    for (unsigned int i = 1; i < header.dimID.size (); i++)
    {
      pt[i] = st.nextFloatToken ();
      points[idx] = pt;
      header.minPD[i] = (pt[i] < header.minPD[i]) ? pt[i] : header.minPD[i];
      header.maxPD[i] = (pt[i] > header.maxPD[i]) ? pt[i] : header.maxPD[i];
    }
    idx++;
    /// ---]
  }
  // Close file
  fs.close ();

  // Compute how much a point (with it's N-dimensions) occupies in terms of bytes
  computePointSize (header);

  /// ---[ Binary mode only
  /// We must re-open the file and read with mmap () for binary
  if (header.data_type == PCD_BINARY)
  {
    //fprintf (stderr, "Binary mode not tested yet in std::vector mode!\n");
    //return (points);
    // Open for reading
    int fd = open (fileName, O_RDONLY);
    if (fd == -1)
      return (points);

    // Compute how much a point (with it's N-dimensions) occupies in terms of bytes
    //computePointSize (header);
    header.point_size = sizeof (float) * header.dimID.size ();

    // Prepare the map
    char *map = (char*)mmap (0, header.nr_points * header.point_size, PROT_READ, MAP_SHARED, fd, getpagesize ());
    if (map == MAP_FAILED)
    {
      close (fd);
      return (points);
    }

    // Prepare the dimensions sizes
    std::vector<int> dimsizes (header.dimID.size ());
    for (unsigned int j = 0; j < header.dimID.size (); j++)
      //dimsizes[j] = computeDimSize (header.dimID[j].second);
      dimsizes[j] = sizeof (float);

    // Read the data
    for (int i = 0; i < header.nr_points; i++)
    {
      int idx_j = 0;
      points[i] = annAllocPt (header.dimID.size ());
      for (unsigned int j = 0; j < header.dimID.size (); j++)
      {
        memcpy (reinterpret_cast<ANNcoord*>(&points[i][j]), (char*)&map[i*header.point_size + idx_j], dimsizes[j]);
        idx_j += dimsizes[j];
        header.minPD[j] = (points[i][j] < header.minPD[j]) ? points[i][j] : header.minPD[j];
        header.maxPD[j] = (points[i][j] > header.maxPD[j]) ? points[i][j] : header.maxPD[j];
      }
    }

    // Unmap the pages of memory
    if (munmap (map, header.nr_points * header.point_size) == -1)
    {
      close (fd);
      return (points);
    }
    close (fd);
  }

  if ((idx != header.nr_points) && (header.data_type == PCD_ASCII))
  {
    fprintf (stderr, "Warning! Number of points read (");
    fprintf (stderr, "%d", idx); fprintf (stderr, ") is different than expected (");
    fprintf (stderr, "%d", header.nr_points); fprintf (stderr, ")!\n");
    header.nr_points = idx;
  }

  // Check for duplicate dimensions
  std::vector<int> c = checkForColumnDuplicates (header);
  if (c.size () != 0)
  {
    fprintf (stderr, "The following columns have a duplicate entry: ");
    for (unsigned int d = 0; d < c.size (); d++)
      fprintf (stderr, "%d(%s) ", header.dimID.at (c.at (d)).first, header.dimID.at (c.at (d)).second.c_str ());
    fprintf (stderr, "\n");
  }
  return points;
}

////////////////////////////////////////////////////////////////////////////////
// Check for special comments in the header of the file that we're loading and
// populate our header.comments with them
void
  readComments (std::string line, PCD_Header &header)
{
  header.comments += line + std::string ("\n");
/*  StringTokenizer st = StringTokenizer (line, "[");
  st.nextToken ();
  std::string comment = st.nextToken ();

  st = StringTokenizer (comment, "]");
  std::string commentType = st.nextToken ();
  std::string commentVal  = st.nextToken ();
  
  // Get the comment type
  if (commentType.substr (0, 8) == "MetaInfo")
    header.comments += line + std::string ("\n");*/
}

// Check for duplicate columns
std::vector<int>
  checkForColumnDuplicates (PCD_Header header)
{
  std::vector<int> idx;
  for (unsigned int d = 0; d < header.dimID.size (); d++)
  {
    std::pair<int, std::string> cur_entry = header.dimID.at (d);
    if (std::count_if (header.dimID.begin (), header.dimID.end (), DimIDComp (cur_entry.second)) > 1)
      idx.push_back (d);
  }
  return idx;
}

////////////////////////////////////////////////////////////////////////////////
// Save a boring n-D histogram file (Note: .hist files have no header, so they 
// can be straightforwardly plotted with gnuplot & co)
int
  SaveHistFile (const char* fileName, ANNpointArray histograms, int nr_samples, int nr_dimensions, int precision)
{
  std::ofstream fs;
  // Open file
  fs.precision (precision);
  fs.open (fileName);

  int nrld = 0;
  // Iterate through the histograms
  for (int cp = 0; cp < nr_samples; cp++)
  {
    for (int d = 0; d < nr_dimensions; d++)
      fs << histograms[cp][d] << " ";
    fs << std::endl;

  }
  // Close file
  fs.close ();
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Load a boring n-D histogram file (Note: .hist files have no header, so they 
// can be straightforwardly plotted with gnuplot & co)
ANNpointArray
  LoadHistFile (const char* fileName, int nr_dimensions, int &nr_samples)
{
  std::ifstream fs;
  std::string line;

  int nrld = 0;
  // Open file
  fs.open (fileName);
  if (!fs.is_open () || fs.fail ())
  {
    fprintf (stderr, "Couldn't open %s for reading!\n", fileName);
    return NULL;
  }
  
  // Get the number of lines (points) this histogram file holds
  nr_samples = 0;
  while ((!fs.eof ()) && (getline (fs, line))) nr_samples++;
  
  ANNpointArray points = annAllocPts (nr_samples, nr_dimensions);

  fs.clear ();  
  fs.seekg (0, std::ios_base::beg);
  for (int cp = 0; cp < nr_samples; cp++)
  {
    getline (fs, line);
    if (line == "")
      continue;

    StringTokenizer st = StringTokenizer (line, " ");

    for (int d = 0; d < nr_dimensions; d++)
      points[cp][d] = st.nextFloatToken ();
      
    /// ---]
  }
  // Close file
  fs.close ();
  return points;
}

////////////////////////////////////////////////////////////////////////////////
// Save point cloud data to a PCD file containing n-D points
int
  SaveASCIIPCDFile (const char* fileName, ANNpointArray points, PCD_Header header, int precision)
{
  std::ofstream fs;
  // Open file
  fs.precision (precision);
  fs.open (fileName);

  // Enforce ASCII mode
  header.data_type = PCD_ASCII;
  
  // Write the header information
  fs << createNewFieldsHeader (header);

  // Add meta-comments
  if (header.comments != "")
    fs << header.comments;

  std::string header_values = addCurrentHeader (header);
  if (header_values == "")
  {
    fprintf (stderr, "Error assembling header! Possible reasons: no FIELDS indices, or no POINTS\n");
    return (-1);
  }
  fs << header_values;

  int nrld = 0;
  // Iterate through the points
  for (int i = 0; i < header.nr_points; i++)
  {
    unsigned int j;
    for (j = 0; j < header.dimID.size () - 1; j++)
      fs << points[i][j] << " ";
    fs << points[i][j] << std::endl;

  }
  // Close file
  fs.close ();
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Save point cloud data to a PCD file containing n-D points given by a vector
// of indices
int
  SaveASCIIPCDFile (const char* fileName, ANNpointArray points, std::vector<int> indices,
                    PCD_Header header, int precision)
{
  std::ofstream fs;
  // Open file
  fs.precision (precision);
  fs.open (fileName);

  header.nr_points = indices.size ();
  // Enforce ASCII mode
  header.data_type = PCD_ASCII;
  
  // Write the header information
  fs << createNewFieldsHeader (header);

  // Add meta-comments
  if (header.comments != "")
    fs << header.comments;

  std::string header_values = addCurrentHeader (header);
  if (header_values == "")
  {
    fprintf (stderr, "Error assembling header! Possible reasons: no FIELDS indices, or no POINTS\n");
    return (-1);
  }
  fs << header_values;

  int nrld = 0;
  // Iterate through the points
  for (unsigned int i = 0; i < indices.size (); i++)
  {
    unsigned int j;
    for (j = 0; j < header.dimID.size () - 1; j++)
      fs << points[indices.at (i)][j] << " ";
    fs << points[indices.at (i)][j] << std::endl;

  }
  // Close file
  fs.close ();
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Save point cloud data to a PCD file containing n-D points given by a vector
// of indices
int
  SaveASCIIPCDFile (const char* fileName, ANNpointArray points, std::set<int> indices,
                    PCD_Header header, int precision)
{
  std::ofstream fs;
  // Open file
  fs.precision (precision);
  fs.open (fileName);

  header.nr_points = indices.size ();
  // Enforce ASCII mode
  header.data_type = PCD_ASCII;
  
  // Write the header information
  fs << createNewFieldsHeader (header);

  // Add meta-comments
  if (header.comments != "")
    fs << header.comments;

  std::string header_values = addCurrentHeader (header);
  if (header_values == "")
  {
    fprintf (stderr, "Error assembling header! Possible reasons: no FIELDS indices, or no POINTS\n");
    return (-1);
  }
  fs << header_values;

  int nrld = 0, i = 0;
  // Iterate through the points
  for (std::set<int>::iterator it = indices.begin (); it != indices.end (); it++)
  {
    unsigned int j;
    for (j = 0; j < header.dimID.size () - 1; j++)
      fs << points[*it][j] << " ";
    fs << points[*it][j] << std::endl;

    i++;
  }
  // Close file
  fs.close ();
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Save point cloud data to a PCD file containing n-D points in binary format
/// @note: using mmap () to speed the s*hit out of the I/O processContents
/// @warning: might be incompatible between different operating systems!
int
  SaveBinaryPCDFile (const char* fileName, ANNpointArray points, PCD_Header header)
{
  int data_idx = 0;
  std::ofstream fs;
  // Open file
  fs.open (fileName);

  // Enforce binary mode
  header.data_type = PCD_BINARY;
  
  // Write the header information
  fs << createNewFieldsHeader (header);

  // Add meta-comments
  if (header.comments != "")
    fs << header.comments;

  std::string header_values = addCurrentHeader (header);
  if (header_values == "")
  {
    fprintf (stderr, "Error assembling header! Possible reasons: no FIELDS indices, or no POINTS\n");
    return (-1);
  }
  fs << header_values;
  data_idx = fs.tellp ();
  // Close file
  fs.close ();

  if (data_idx > getpagesize ())
  {
    fprintf (stderr, "Warning: header size (%d) is bigger than page size (%d)! Are you sure you need that many bytes? Removing extra comments...\n", data_idx, getpagesize ());
    
    fs.open (fileName);
    header.data_type = PCD_BINARY;
    fs << createNewHeaderLight ();
    if (header.comments != "")
      fs << header.comments;
    std::string header_values = addCurrentHeader (header);
    if (header_values == "")
    {
      fprintf (stderr, "Error assembling header! Possible reasons: no FIELDS indices, or no POINTS\n");
      return (-1);
    }
    fs << header_values;
    data_idx = fs.tellp ();
    // Close file
    fs.close ();
    if (data_idx > getpagesize ())
      fprintf (stderr, "Error: header size (%d) is *still* bigger than page size (%d) even after I removed the extra comments! Please consider removing all coments, or save file as ASCII.\n", data_idx, getpagesize ());
  }

  // Compute how much a point (with it's N-dimensions) occupies in terms of bytes
  //computePointSize (header);
  header.point_size = sizeof (float) * header.dimID.size ();

  // Prepare the dimensions sizes
  std::vector<int> dimsizes (header.dimID.size ());
  for (unsigned int j = 0; j < header.dimID.size (); j++)
    //dimsizes[j] = computeDimSize (header.dimID[j].second);
    dimsizes[j] = sizeof (float);

  // Open for writing
  int fd = open (fileName, O_RDWR);
  if (fd == -1)
  {
    fprintf (stderr, "Error during open ()!\n");
    return (-1);
  }

  // Stretch the file size to the size of the data
  int result = lseek (fd, getpagesize () + (header.nr_points * header.point_size) - 1, SEEK_SET);
  if (result == -1)
  {
    close (fd);
    fprintf (stderr, "Error during lseek ()!\n");
    return (-1);
  }
  // Write a bogus entry so that the new file size comes in effect
  result = write (fd, "", 1);
  if (result != 1)
  {
    close (fd);
    fprintf (stderr, "Error during write ()!\n");
    return (-1);
  }

  // Prepare the map
  char *map = (char*)mmap (0, header.nr_points * header.point_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, getpagesize ());
  if (map == MAP_FAILED)
  {
    close (fd);
    fprintf (stderr, "Error during mmap ()!\n");
    return (-1);
  }

  // Write the data
  int nrld = 0;
  for (int i = 0; i < header.nr_points; i++)
  {
    int idx_j = 0;
    for (unsigned int j = 0; j < header.dimID.size (); j++)
    {
      memcpy ((char*)&map[i*header.point_size + idx_j], reinterpret_cast<char*>(&points[i][j]), dimsizes[j]);
      idx_j += dimsizes[j];
    }
  }

  // Unmap the pages of memory
  if (munmap (map, (header.nr_points * header.point_size)) == -1)
  {
    close (fd);
    fprintf (stderr, "Error during munmap ()!\n");
    return (-1);
  }

  // Close file
  close (fd);
  return (0);
}

////////////////////////////////////////////////////////////////////////////////
// Save point cloud data to a PCD file containing n-D points
int
  SavePCDFile (const char* fileName, ANNpointArray points, PCD_Header header, int precision)
{
  if (header.data_type == PCD_BINARY)
    return (SaveBinaryPCDFile (fileName, points, header));
  else
    return (SaveASCIIPCDFile (fileName, points, header, precision));
}

////////////////////////////////////////////////////////////////////////////////
// Save point cloud data to a PCD file containing n-D points
int
  SavePCDFile (const char* fileName, std::vector<ANNpoint> points, PCD_Header header, int precision)
{
  return SavePCDFile (fileName, reinterpret_cast <ANNpointArray>(&points[0]), header, precision);
}

////////////////////////////////////////////////////////////////////////////////
// We add this string to every new PCD file we create. 'Just common sense, mkay?
std::string
  createNewHeader ()
{
  return std::string (
    "# .PCD v.5 - Point Cloud Data file format\n"
    "#\n"
/*    "# Column legend:\n"
    "#   x,  y,  z - 3d point coordinates\n"
    "#  nx, ny, nz - point normals\n"
    "#  vx, vy, vz - view point information for current point\n"
    "#   r,  g,  b - RGB point color information\n"
    "#           c - curvature estimation (A_min / sum (A)), A - eigenvalues\n"
    "#          bp - boundary point property (0 / 1)\n"
    "#          cc - local curvature estimation (cc = 2*d/u - [see Mitra - Estimating Surface Normals in Noisy Poiny Cloud Data])\n"
    "#         ivd - Integral Volume Descriptor (Vr = (2*pi*r^3)/3 + (pi*H*r^4)/4 - [see Gelfand - Robust Global Registration], H = mean curvature)\n"
    "#           k - cluster size (adaptive or radius-based), equivalent with the bucket density (the number of points in a certain region)\n"
    "#           i - laser intensity/remission information\n"
    "#           d - distance from the laser viewpoint to the surface\n"
    "#         sid - laser scan ID (eg. the N'th measurement in one laser packet)\n"
    "#         pid - laser packet ID (eg. the ID of the N'th laser packet from the sensor)\n"
    "#         reg - region number ID (after segmentation)\n"
    "#          oc - object class ID (see common/ObjectClasses.h)\n"
    "#          pf - persistent feature\n"
    "#          sp - Special Point -- this should be a placeholder used by any routine which works with some preselected points,\n"
    "#               like, say I want to mark only the points on the edge of a cube for additional feature estimation.\n"
    "# NOTE: columns ca be swapped and arranged in any order (almost*) as long as the indices names are swapped too.\n"
    "#    *= the assumption is that groups of coordinates (eg. x-y-z, nx-ny-nz, r-g-b) are ordered consecutively.\n"
    "#\n"
    "# DATA specifies whether points are stored in binary format or ascii text. Header must be page aligned (4096 on most systems)\n#\n*"*/);
}

////////////////////////////////////////////////////////////////////////////////
std::string
  createNewFieldsHeader (PCD_Header header)
{
  std::string result = "# .PCD v.5 - Point Cloud Data file format\n#\nFIELDS ";

  result += getAvailableDimensions (header);

  result += "\n";

  // Must have both the FIELDS order as well as the number of POINTS
  if ((header.dimID.size () == 0) || (header.nr_points == 0))
    return ("");

  return (result);
}

////////////////////////////////////////////////////////////////////////////////
// We add this string to every new PCD file we create. 'Just common sense, mkay?
std::string
  createNewHeaderLight ()
{
  return std::string (
    "# .PCD v.5 - Point Cloud Data file format\n"
    "#\n"
/*    "# DATA specifies whether points are stored in binary format or ascii text. Header must be page aligned (4096 on most systems)\n#\n"*/);
}

////////////////////////////////////////////////////////////////////////////////
// Add a comment to the PCD header (maximum size: 256 characters)
void
  addCommentToHeader (PCD_Header &header, const char *format, ...)
{
  char str[256];

  va_list ap;

  va_start (ap, format);
  vsprintf (str, format, ap);
  va_end (ap);

  header.comments += std::string ("# [MetaInfo]: ") + std::string (str);
}

////////////////////////////////////////////////////////////////////////////////
// Get the available dimensions as a space separated string
std::string
  getAvailableDimensions (PCD_Header header)
{
  std::string result;
  unsigned int i;
  for (i = 0; i < header.dimID.size () - 1; i++)
  {
    if (header.dimID[i].first != -1)
    {
      std::string index = header.dimID[i].second + " ";
      result += index;
    }
  }
  if (header.dimID[i].first != -1)
  {
    std::string index = header.dimID[i].second;
    result += index;
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Create the rest of the needed string values from the given header
std::string
  addCurrentHeader (PCD_Header header)
{
/*  std::string result = "FIELDS ";

  result += getAvailableDimensions (header);

  // Must have both the FIELDS order as well as the number of POINTS
  if ((header.dimID.size () == 0) || (header.nr_points == 0))
    return "";
*/
  std::string result;
  std::ostringstream oss;
  oss << result << "POINTS " << header.nr_points << "\n";

  oss << "DATA ";
  if (header.data_type == PCD_BINARY)
    oss << "binary\n";
  else
    oss << "ascii\n";

  result = oss.str ();
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Get the column index of a specified dimension
int
  getIndex (PCD_Header header, std::string value)
{
  // Get the indices we need automatically
  int x_idx;
  std::vector<std::pair<int, std::string> >::iterator result_x;
  result_x = std::find_if (header.dimID.begin (), header.dimID.end (), DimIDComp (value));
  if (result_x == header.dimID.end ())
    return (-1);
  x_idx = (*result_x).first;
  return (x_idx);
}

////////////////////////////////////////////////////////////////////////////////
// Get the maximum column index for a dimension starting with 'value'
int
  getMaximumFeatureDimension (PCD_Header header, std::string value)
{
  std::set<int> idx;
  std::vector<std::pair<int, std::string> >::iterator it;
  // Iterate over all dimensions
  for (it = header.dimID.begin (); it != header.dimID.end (); ++it)
  {
    std::pair<int, std::string> d = *it;
    // Find the location of the value string in the current dimension string
    std::string::size_type loc = d.second.find (value.c_str (), 0);
    if (loc != std::string::npos)
    {
      int s = (d.second.size () - value.size ());
      char val[s+1];
      memset (val, '\0', s+1);
      d.second.copy (val, s, loc + value.size ());
      idx.insert (strtol (val, NULL, 0));
    }
  }
  if (idx.size () > 0)
  {
    std::set<int>::iterator i;
    i = idx.end ();
    i--;
    return (*i);
  }
  else
    return (0);
}

////////////////////////////////////////////////////////////////////////////////
// Get the memory size occupied by this dimension
/// @note: any better ideas on how to do this efficiently ?
int
  computeDimSize (std::string value)
{
  if (                            // <x, y, z, nx, ny, nz, c, cc, ivd, d> are all floats/doubles
      (value == "x") ||
      (value == "y") ||
      (value == "z") ||
      (value == "nx") ||
      (value == "ny") ||
      (value == "nz") ||
      (value == "vx") ||
      (value == "vy") ||
      (value == "vz") ||
      (value == "c") ||
      (value == "cc") ||
      (value == "ivd") ||
      (value == "sp") ||
      (value == "d"))
    return (sizeof (float));
  
  if (                            // <r, g, b, pf, bp> are unsigned chars (0..255)
      (value == "r") ||
      (value == "g") ||
      (value == "b") ||
      (value == "pf") ||
      (value == "bp"))
    return (sizeof (unsigned char));
    //return (sizeof (short));

  if (                            // <k, i, sid, pid, reg> are all integers
      (value == "k") ||
      (value == "i") ||
      (value == "sid") ||
      (value == "pid") ||
      (value == "reg") ||
      (value == "oc"))
    return (sizeof (int));
  
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Parse a dimension argument (as string), check its existence, and return its
// index (if it exists)
int
  ParseDimensionArgument (int argc, char** argv, const char* str, std::string &val, PCD_Header header)
{
  int idx;
  for (int i = 1; i < argc; i++)
  {
    // Search for the string
    if ((strcmp (argv[i], str) == 0) && (++i < argc))
    {
      val = std::string (argv[i]);
      if ((idx = getIndex (header, val)) != -1)
        return (idx);
      else
        return (-1);
    }
  }
  return (-1);
}

////////////////////////////////////////////////////////////////////////////////
// Compute the memory size (in bytes) occupied by a single point in the array
void
  computePointSize (PCD_Header &header)
{
  header.point_size = 0;
  std::vector<std::pair<int, std::string> >::iterator result_x;
  for (unsigned int i = 0; i < header.dimID.size (); i++)
  {
    std::string value = header.dimID[i].second;
    result_x = std::find_if (header.dimID.begin (), header.dimID.end (), DimIDComp (value));
    if (result_x != header.dimID.end ())
      header.point_size += (int)computeDimSize (value);
  }
}
