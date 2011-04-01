
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


int main(int argc, char *argv[])
{
  test_load_pcd(argc, argv);

  return 0;
}
