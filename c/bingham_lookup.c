
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/bingham_constants.h"




void usage(int argc, char *argv[])
{
  printf("usage:  bingham_lookup <z1> <z2> <z3>\n");
  exit(1);
}

int main(int argc, char *argv[])
{
  if (argc < 4)
    usage(argc, argv);

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);
  double Z[3] = {z1, z2, z3};

  double F = bingham_F_lookup_3d(Z);
  double dF[3];
  bingham_dF_lookup_3d(dF, Z);

  printf("F = %e, dF = [%e, %e, %e]\n", F, dF[0], dF[1], dF[2]);

  return 0;
}
