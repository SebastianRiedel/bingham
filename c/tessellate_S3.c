#include <stdio.h>
#include "bingham/hypersphere.h"
#include "bingham/util.h"


int main(int argc, char *argv[])
{
  if (argc < 2) {
    printf("usage: %s <num points>\n", argv[0]);
    return 1;
  }

  hypersphere_tessellation_t *T = tessellate_S3(atoi(argv[1]));

  int i,j;

  printf("%d %d\n", T->n, T->d + 1);
  for (i = 0; i < T->n; i++) {
    for (j = 0; j < T->d; j++)
      printf("%f ", T->centroids[i][j]);
    printf("%f\n", T->volumes[i]);
  }

  return 0;
}
