
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "bingham/tetramesh.h"
#include "bingham/octetramesh.h"
#include "bingham/hypersphere.h"
#include "bingham/util.h"


tetramesh_t *simplex3d()
{
  tetramesh_t *T;
  safe_calloc(T, 1, tetramesh_t);
  tetramesh_new(T, 4, 1, 3);

  double *p0 = T->vertices[0];
  double *p1 = T->vertices[1];
  double *p2 = T->vertices[2];
  double *p3 = T->vertices[3];

  p0[0] = 1;  p0[1] = 0;  p0[2] = 0;
  p1[0] = 0;  p1[1] = 1;  p1[2] = 0;
  p2[0] = 0;  p2[1] = 0;  p2[2] = 1;
  p3[0] = 0;  p3[1] = 0;  p3[2] = 0;

  int *t = T->tetrahedra[0];
  t[0] = 0;  t[1] = 1;  t[2] = 2;  t[3] = 3;

  return T;
}


tetramesh_t *simplex4d()
{
  tetramesh_t *T;
  safe_calloc(T, 1, tetramesh_t);
  tetramesh_new(T, 4, 1, 4);

  double *p0 = T->vertices[0];
  double *p1 = T->vertices[1];
  double *p2 = T->vertices[2];
  double *p3 = T->vertices[3];

  p0[0] = 1;  p0[1] = 0;  p0[2] = 0;  p0[3] = 0;
  p1[0] = 0;  p1[1] = 1;  p1[2] = 0;  p1[3] = 0;
  p2[0] = 0;  p2[1] = 0;  p2[2] = 1;  p2[3] = 0;
  p3[0] = 0;  p3[1] = 0;  p3[2] = 0;  p3[3] = 1;

  int *t = T->tetrahedra[0];
  t[0] = 0;  t[1] = 1;  t[2] = 2;  t[3] = 3;

  return T;
}


void print_tetramesh(tetramesh_t *T)
{
  int i, j;
  printf("tetramesh_t {\n");
  printf("  (nv = %d, nt = %d, d = %d)\n", T->nv, T->nt, T->d);
  printf("  vertices:\n");
  for (i = 0; i < T->nv; i++) {
    printf("    (");
    for (j = 0; j < T->d; j++)
      printf("%f, ", T->vertices[i][j]);
    printf(")\n");
  }
  printf("  tetrahedra:\n");
  for (i = 0; i < T->nt; i++)
    printf("    (%d, %d, %d, %d)\n", T->tetrahedra[i][0], T->tetrahedra[i][1], T->tetrahedra[i][2], T->tetrahedra[i][3]);
  printf("}\n");
}


void test_subdivide()
{
  tetramesh_t *T = simplex3d();
  tetramesh_t T2;

  print_tetramesh(T);
  printf("\n*** Subdividing ***\n\n");
  tetramesh_subdivide(&T2, T);
  print_tetramesh(&T2);

  //tetramesh_save_PLY(T, "T.ply");
  //tetramesh_save_PLY(&T2, "T2.ply");

  tetramesh_free(&T2);
  tetramesh_free(T);
  free(T);
}


void test_smooth()
{
  tetramesh_t *T = simplex3d();

  print_tetramesh(T);
  //tetramesh_save_PLY(T, "T.ply");
  printf("\n*** Smoothing ***\n\n");
  tetramesh_smooth(T, T, .5);
  //tetramesh_save_PLY(T, "T2.ply");
  print_tetramesh(T);

  tetramesh_free(T);
  free(T);
}


void check_regularity(tetramesh_t *T)
{
  tetramesh_stats_t stats = tetramesh_stats(T);
  tetramesh_print_stats(stats);
}


void test_hypersphere(int argc, char *argv[])
{
  if (argc < 2) {
    printf("usage: %s <n>\n", argv[0]);
    exit(1);
  }

  int n = atoi(argv[1]);

  octetramesh_t *mesh;
  mesh = hypersphere_tessellation_octetra(n);
  octetramesh_save_PLY(mesh, "mesh.ply");
  octetramesh_free(mesh);
  free(mesh);

  tetramesh_t *T;
  T = hypersphere_tessellation_tetra(n);
  //tetramesh_save_PLY(T, "T.ply");
  tetramesh_free(T);
  free(T);
}


void test_solve()
{
  double A[] = {0, 1, 1,
		1, 1, 0,
		1, 0, 1};
  double b[] = {3, 2, 1};
  double x[3];

  solve(x, A, b, 3);

  printf("x = (%.2f, %.2f, %.2f)\n", x[0], x[1], x[2]);

  assert(x[0] == 0 && x[1] == 2 && x[2] == 1);
}


int main(int argc, char *argv[])
{
  //test_subdivide();
  //test_smooth();
  test_hypersphere(argc, argv);
  //test_solve();

  return 0;
}


