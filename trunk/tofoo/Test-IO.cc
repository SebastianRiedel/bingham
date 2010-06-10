/* $Id: Test-CommonANN.cc,v 1.0 2006/10/04 12:00:00 radu Exp $
 */
#include "CommonIORoutines.h"

using namespace std;

/* ---[ */
int
  main (int argc, char** argv)
{
  PCD_Header h;
  ANNpointArray points = NULL;
  // Load file
  fprintf (stderr, "Loading "); fprintf (stderr, "%s", argv[1]); fprintf (stderr, "... ");
  points = LoadPCDFile (argv[1], h);
  fprintf (stderr, "[done : points in ");
  if (h.data_type == PCD_BINARY) 
    fprintf (stderr, "binary");
  else
    fprintf (stderr, "ascii");
  fprintf (stderr, " mode]\n");  
  
  
  // Save file in the "other" mode/format
  fprintf (stderr, "Saving to... "); fprintf (stderr, "t1.pcd"); 
  h.data_type = !h.data_type;
  fprintf (stderr, " in ");
  if (h.data_type == PCD_BINARY) 
    fprintf (stderr, "binary");
  else
    fprintf (stderr, "ascii");
  fprintf (stderr, " mode... ");  
  SavePCDFile ("t1.pcd", points, h, 5);
  fprintf (stderr, "[done]\n");
  

  // Load file in "other" mode/format
  fprintf (stderr, "Loading "); fprintf (stderr, "t1.pcd"); fprintf (stderr, "... ");
  points = LoadPCDFile ("t1.pcd", h);
  fprintf (stderr, "[done : %d", h.nr_points); fprintf (stderr, " points in ");
  if (h.data_type == PCD_BINARY) 
    fprintf (stderr, "binary");
  else
    fprintf (stderr, "ascii");
  fprintf (stderr, " mode]\n");  
  
  // Save file in the original format
  fprintf (stderr, "Saving to... "); fprintf (stderr, "t2.pcd"); 
  h.data_type = !h.data_type;
  fprintf (stderr, " in ");
  if (h.data_type == PCD_BINARY) 
    fprintf (stderr, "binary");
  else
    fprintf (stderr, "ascii");
  fprintf (stderr, " mode... ");  
  SavePCDFile ("t2.pcd", points, h, 5);
  fprintf (stderr, "[done]\n");
}
/* ]--- */
