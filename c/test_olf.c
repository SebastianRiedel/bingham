
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/olf.h"




void test_load_pcd(int argc, char *argv[])
{
  int i, j;

  if (argc < 2) {
    printf("usage: %s <f_pcd>\n", argv[0]);
    return;
  }

  pcd_t *pcd = load_pcd(argv[1]);

  if (pcd) {

    printf("pcd:\n");
    printf("  num_channels = %d\n", pcd->num_channels);
    printf("  num_points = %d\n", pcd->num_points);
    for (i = 0; i < pcd->num_channels; i++) {
      printf("  %s: ", pcd->channels[i]);
      for (j = 0; j < pcd->num_points; j++)
	printf("%f ", pcd->data[i][j]);
      printf("\n");
    }

    pcd_free(pcd);
    free(pcd);
  }
}


void test_olf_pose_sample(int argc, char *argv[])
{
  if (argc < 4) {
    printf("usage: %s <olf> <pcd> <n>\n", argv[0]);
    return;
  }

  double t = get_time_ms();
  olf_t *olf = load_olf(argv[1]);
  fprintf(stderr, "Loaded olf in %f ms\n", get_time_ms() - t);

  t = get_time_ms();
  pcd_t *pcd = load_pcd(argv[2]);
  fprintf(stderr, "Loaded pcd in %f ms\n", get_time_ms() - t);

  int n = atof(argv[3]);

  if (olf == NULL) {
    printf("Error loading olf\n");
    return;
  }
  if (pcd == NULL) {
    printf("Error loading pcd\n");
    return;
  }
  
  t = get_time_ms();
  olf_classify_points(pcd, olf);
  fprintf(stderr, "Classified local shapes in %f ms\n", get_time_ms() - t);

  olf_pose_samples_t *poses;
  //int xxx;
  //for (xxx = 0; xxx < 10000; xxx++) {
  t = get_time_ms();
  poses = olf_pose_sample(olf, pcd, n);
  fprintf(stderr, "Sampled %d poses in %f ms\n", n, get_time_ms() - t);
  //}

  fprintf(stderr, "W[0] = %f\n", poses->W[0]);

  t = get_time_ms();
  poses = olf_aggregate_pose_samples(poses, olf);
  fprintf(stderr, "Aggregated %d->%d poses in %f ms\n", n, poses->n, get_time_ms() - t);

  fprintf(stderr, "W[0] = %f\n", poses->W[0]);

  double **X = poses->X;
  double **Q = poses->Q;
  double *W = poses->W;
  n = poses->n;

  int i;
  printf("X = [");
  for (i = 0; i < n-1; i++)
    printf("%f, %f, %f; ...\n", X[i][0], X[i][1], X[i][2]);
  printf("%f, %f, %f];\n", X[i][0], X[i][1], X[i][2]);

  printf("Q = [");
  for (i = 0; i < n-1; i++)
    printf("%f, %f, %f, %f; ...\n", Q[i][0], Q[i][1], Q[i][2], Q[i][3]);
  printf("%f, %f, %f, %f];\n", Q[i][0], Q[i][1], Q[i][2], Q[i][3]);

  printf("W = [");
  for (i = 0; i < n-1; i++)
    printf("%f; ...\n", W[i]);
  printf("%f];\n", W[i]);
}


int main(int argc, char *argv[])
{
  //test_load_pcd(argc, argv);
  test_olf_pose_sample(argc, argv);

  return 0;
}
