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

@b Sample points from a bingham mixture model.

 **/



// eigen
#include <Eigen/Core>
#include <Eigen/LU>
USING_PART_OF_NAMESPACE_EIGEN
//#define EIGEN_USE_NEW_STDVECTOR
//#include <Eigen/StdVector>


#include <float.h>

// bingham
extern "C" {
#include <bingham.h>
}

#include <ros/ros.h>


using namespace std;



/*
 * Returns an array of bingham mixtures
 */
bingham_mix_t *getBinghamMixtures(char *f_bmx, int &k)
{
  FILE *f = fopen(f_bmx, "r");

  if (f == NULL) {
    ROS_ERROR("Invalid filename: %s", f_bmx);
    exit(1);
  }

  // get the number of binghams mixtures in the bmx file
  k = 0;
  char sbuf[1024], *s = sbuf;
  int c;
  while (!feof(f)) {
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d", &c) && c+1 > k)
      k = c+1;
  }
  rewind(f);

  bingham_mix_t *BM = (bingham_mix_t *)calloc(k, sizeof(bingham_mix_t));

  // get the number of binghams in each mixture
  int i;
  while (!feof(f)) {
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d %d", &c, &i) == 2 && i+1 > BM[c].n)
      BM[c].n = i+1;
  }
  rewind(f);

  // allocate space for the binghams
  for (c = 0; c < k; c++) {
    BM[c].w = (double *)calloc(BM[c].n, sizeof(double));
    BM[c].B = (bingham_t *)calloc(BM[c].n, sizeof(bingham_t));
  }

  // read in the binghams and corresponding weights
  int d, j, j2;
  double w;
  int line = 0;
  while (!feof(f)) {
    line++;
    s = sbuf;
    fgets(s, 1024, f);
    if (s[0] == 'B' && sscanf(s, "B %d %d %lf %d", &c, &i, &w, &d) == 4) {
      BM[c].w[i] = w;
      BM[c].B[i].d = d;
      BM[c].B[i].Z = (double *)calloc(d-1, sizeof(double));
      BM[c].B[i].V = new_matrix2(d-1, d);
      s = sword(s, " \t", 5);
      if (sscanf(s, "%lf", &BM[c].B[i].F) < 1)  // read F
	break;
      s = sword(s, " \t", 1);
      for (j = 0; j < d-1; j++) {  // read Z
	if (sscanf(s, "%lf", &BM[c].B[i].Z[j]) < 1)
	  break;
	s = sword(s, " \t", 1);
      }
      if (j < d-1)  // error
	break;
      for (j = 0; j < d-1; j++) {  // read V
	for (j2 = 0; j2 < d; j2++) {
	  if (sscanf(s, "%lf", &BM[c].B[i].V[j][j2]) < 1)
	    break;
	  s = sword(s, " \t", 1);
	}
	if (j2 < d)  // error
	  break;
      }
      if (j < d-1)  // error
	break;
    }
  }
  if (!feof(f)) {  // error
    fprintf(stderr, "Error reading file %s at line %d.\n", f_bmx, line);
    return NULL;
  }
  fclose(f);

  return BM;
}



void sampleBMX(char *f_bmx)
{
  int num_bmix;
  bingham_mix_t *BM = getBinghamMixtures(f_bmx, num_bmix);



  // return the peaks
  
  /*
  int n = BM->n;
  int d = BM->B[0].d;

  double peaks[n][d];
  double pdf[n];
  int idx[n];
  bingham_stats_t stats;
  for (int i = 0; i < n; i++) {
    bingham_stats(&stats, &BM->B[i]);
    for (int j = 0; j < d; j++)
      peaks[i][j] = stats.mode[j];
    pdf[i] = -bingham_mixture_pdf(stats.mode, BM);
  }
  mult(pdf, pdf, -1/min(pdf, n), n);  //dbug
  sort_indices(pdf, idx, n);

  printf("\n\nX = [ ...\n");
  for (int j = 0; j < n; j++)
    printf("%f, %f, %f, %f ; ...\n", peaks[idx[j]][0], peaks[idx[j]][1], peaks[idx[j]][2], peaks[idx[j]][3]);
  printf("];\n\n\n");
  printf("pdf = [ ");
  for (int j = 0; j < n; j++)
    printf("%f ", -pdf[idx[j]]);
  printf("];\n\n");
  */
  



  //---------- ridge search ----------//

  /*
  int nsamples = 50;
  double **X = new_matrix2(nsamples, 4);
  bingham_mixture_sample_ridge(X, BM, nsamples, 1/surface_area_sphere(3)); //.1*max_peak);
  printf("\n\nX = [ ...\n");
  for (int j = 0; j < nsamples; j++)
    printf("%f, %f, %f, %f ; ...\n", X[j][0], X[j][1], X[j][2], X[j][3]);
  printf("];\n\n\n");
  free_matrix2(X);
  */


  //---------- tessellation search ----------//

  int nsamples = 50;
  bingham_pmf_t pmf;
  bingham_discretize(&pmf, &BM[0].B[0], 10000);
  double **R = pmf.points;
  int nr = 0;
  for (int i = 0; i < pmf.n; i++) {  // only need to keep one hemisphere of S3
    if (R[i][0] >= 0) {
      R[nr][0] = R[i][0];
      R[nr][1] = R[i][1];
      R[nr][2] = R[i][2];
      R[nr][3] = R[i][3];
      nr++;
    }
  }

  //dbug
  R[0][0] = 0;
  R[0][1] = 0;
  R[0][2] = 0;
  R[0][3] = 1;

  int *indices = (int *)calloc(nr, sizeof(int));
  double *pdf = (double *)calloc(nr, sizeof(double));
  for (int i = 0; i < nr; i++)
    pdf[i] = -bingham_mixture_pdf(R[i], BM);
  mult(pdf, pdf, -1/min(pdf, nr), nr);  //dbug
  sort_indices(pdf, indices, nr);
  printf("\n\nX = [ ...\n");
  for (int j = 0; j < nsamples; j++)
    printf("%f, %f, %f, %f ; ...\n", R[indices[j]][0], R[indices[j]][1], R[indices[j]][2], R[indices[j]][3]);
  printf("];\n\n\n");
  printf("pdf = [ ");
  for (int j = 0; j < nsamples; j++)
    printf("%f ", -pdf[indices[j]]);
  printf("]\n\n");
  free(indices);
  free(pdf);
}


void usage(int argc, char **argv)
{
  ROS_ERROR("usage: %s <bmx>", argv[0]);
  exit(1);
}


int main(int argc, char** argv)
{
  if (argc < 2)
    usage(argc, argv);

  char *f_bmx = argv[1];

  sampleBMX(f_bmx);

  return 0;
}
