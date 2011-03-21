
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "bingham/util.h"
#include "bingham/olf.h"



/*
 * loads a pcd
 */
pcd_t *load_pcd(char *f_pcd)
{
  FILE *f = fopen(f_pcd, "r");

  if (f == NULL) {
    fprintf(stderr, "Invalid filename: %s", f_pcd);
    return NULL;
  }

  pcd_t *pcd;
  safe_calloc(pcd, 1, pcd_t);

  char sbuf[1024], *s = sbuf;
  while (!feof(f)) {
    fgets(s, 1024, f);
    
    if (!wordcmp(s, "COLUMNS", " \t\n") || !wordcmp(s, "FIELDS", " \t\n")) {
      s = sword(s, " \t", 1);
      pcd->channels = split(s, " \t", &pcd->num_channels);
    }
    else if (!wordcmp(s, "POINTS", " \t\n")) {
      s = sword(s, " \t", 1);
      sscanf(s, "%d", &pcd->num_points);
    }
    else if (!wordcmp(s, "DATA", " \t\n")) {
      s = sword(s, " \t", 1);
      if (wordcmp(s, "ascii", " \t\n")) {
	fprintf(stderr, "Error: only ascii pcd files are supported.\n");
	pcd_free(pcd);
	free(pcd);
	return NULL;
      }
      pcd->data = new_matrix2(pcd->num_channels, pcd->num_points);  // load channels in the rows
      int i, j;
      for (i = 0; i < pcd->num_points; i++) {
	if (fgets(s, 1024, f) == NULL)
	  break;
	for (j = 0; j < pcd->num_channels; j++) {
	  if (sscanf(s, "%lf", &pcd->data[j][i]) < 1)
	    break;
	  s = sword(s, " \t", 1);
	}
	if (j < pcd->num_channels)
	  break;
      }
      if (i < pcd->num_points) {
	fprintf(stderr, "Error: corrupt pcd data at row %d\n", i);
	pcd_free(pcd);
	free(pcd);
	return NULL;
      }
    }
  }

  return pcd;
}


/*
 * frees the contents of a pcd_t, but not the pointer itself
 */
void pcd_free(pcd_t *pcd)
{
  if (pcd == NULL)
    return;

  if (pcd->channels) {
    int i;
    for (i = 0; i < pcd->num_channels; i++)
      if (pcd->channels[i])
	free(pcd->channels[i]);
    free(pcd->channels);
  }

  if (pcd->data)
    free_matrix2(pcd->data);
}


/*
 * loads an olf from fname.pcd and fname.olf
 */
olf_t *load_olf(char *fname)
{

}


/*
 * frees the contents of an olf_t, but not the pointer itself
 */
void olf_free(olf_t *olf)
{

}

