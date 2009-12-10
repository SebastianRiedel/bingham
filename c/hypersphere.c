
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "bingham/util.h"
#include "bingham/hypersphere.h"



// cached tessellations
#define MAX_LEVELS 20
tetramesh_t *tessellations[MAX_LEVELS];



/*
 * Reproject mesh vertices onto the unit hypersphere.
 */
static void reproject_vertices(double **vertices, int nv, int dim)
{
  int i;
  for (i = 0; i < nv; i++) {
    double d = norm(vertices[i], dim);
    mult(vertices[i], vertices[i], 1/d, dim);
    //printf("reprojected vertex %d: %f -> %f\n", i, d, norm(vertices[i], dim));
  }
}


/*
 * Create the initial (low-res) mesh of S3 (in R4).
 */
static octetramesh_t *init_mesh_S3_octetra()
{
  octetramesh_t *mesh;
  safe_calloc(mesh, 1, octetramesh_t);
  octetramesh_new(mesh, 8, 16, 0, 4);

  double vertices[8][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1},
			   {-1,0,0,0}, {0,-1,0,0}, {0,0,-1,0}, {0,0,0,-1}};

  int tetrahedra[16][4] = {{0,1,2,3}, {0,1,2,7}, {0,1,6,3}, {0,5,2,3},
			   {0,1,6,7}, {0,5,2,7}, {0,5,6,3}, {0,5,6,7},
			   {4,1,2,3}, {4,1,2,7}, {4,1,6,3}, {4,5,2,3},
			   {4,1,6,7}, {4,5,2,7}, {4,5,6,3}, {4,5,6,7}};

  memcpy(mesh->vertices[0], vertices, 8*4*sizeof(double));
  memcpy(mesh->tetrahedra[0], tetrahedra, 16*4*sizeof(int));

  return mesh;
}


/*
 * Create a tesselation of the 3-sphere (in R4) at a given level (# of subdivisions)
 */
static octetramesh_t *build_octetra(int level)
{
  int i;
  octetramesh_t *mesh = init_mesh_S3_octetra();
  octetramesh_t tmp;

  for (i = 0; i < level; i++) {
    octetramesh_subdivide(&tmp, mesh);
    octetramesh_free(mesh);
    free(mesh);
    mesh = octetramesh_clone(&tmp);
    octetramesh_free(&tmp);
  }

  reproject_vertices(mesh->vertices, mesh->nv, mesh->d);

  return mesh;
}

/*
 * Pre-cache some tessellations of hyperspheres.
 */
void hypersphere_init()
{
  double t0 = get_time_ms();

  int i;
  const int levels = 7;

  memset(tessellations, 0, MAX_LEVELS*sizeof(tetramesh_t *));

  for (i = 0; i < levels; i++) {
    octetramesh_t *mesh = build_octetra(i);
    tessellations[i] = octetramesh_to_tetramesh(mesh);
  }

  printf("Initialized %d hypersphere tessellations (up to %d cells) in %.0f ms\n", levels, tessellations[levels-1]->nt, get_time_ms() - t0);
}


/*
 * Returns a tesselation of the 3-sphere (in R4) with at least n cells.
 */
tetramesh_t *tessellate_S3(int n)
{
  int i;
  for (i = 0; i < MAX_LEVELS; i++) {
    if (tessellations[i] == NULL)
      break;
    if (tessellations[i]->nt >= n)
      return tessellations[i];
  }

  if (i < MAX_LEVELS) {
    octetramesh_t *mesh = build_octetra(i);
    tessellations[i] = octetramesh_to_tetramesh(mesh);
    return tessellations[i];
  }

  return NULL;
}







//-------------------- DEPRECATED ------------------//


/*
 * Create the initial (low-res) mesh of S3 (in R4).
 *
static tetramesh_t *init_mesh_S3_tetra()
{
  tetramesh_t *T;
  safe_calloc(T, 1, tetramesh_t);
  tetramesh_new(T, 8, 16, 4);

  double vertices[8][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1},
			   {-1,0,0,0}, {0,-1,0,0}, {0,0,-1,0}, {0,0,0,-1}};

  int tetrahedra[16][4] = {{0,1,2,3}, {0,1,2,7}, {0,1,6,3}, {0,5,2,3},
			   {0,1,6,7}, {0,5,2,7}, {0,5,6,3}, {0,5,6,7},
			   {4,1,2,3}, {4,1,2,7}, {4,1,6,3}, {4,5,2,3},
			   {4,1,6,7}, {4,5,2,7}, {4,5,6,3}, {4,5,6,7}};

  memcpy(T->vertices[0], vertices, 8*4*sizeof(double));
  memcpy(T->tetrahedra[0], tetrahedra, 16*4*sizeof(int));

  return T;
}
*/

/*
 * Create a tesselation of the 3-sphere (in R4) with at least n cells.
 *
tetramesh_t *hypersphere_tessellation_tetra(int n)
{
  int i;
  tetramesh_t *T = init_mesh_S3_tetra();
  tetramesh_t tmp;

  // div = k s.t. nt*8^k > n  ==> 8^k > n/nt
  int div = (int) ceil(log(n/(double)T->nt) / log(8));  // number of times to subdivide
  div = MAX(div, 0);

  for (i = 0; i < div; i++) {

    tetramesh_subdivide(&tmp, T);
    printf("\nSubdivision %d...\n", i+1);

    tetramesh_free(T);
    free(T);
    T = tetramesh_clone(&tmp);

    tetramesh_free(&tmp);
  }

  reproject_vertices(T->vertices, T->nv, T->d);
  printf("\nProjection...\n");
  tetramesh_print_stats(tetramesh_stats(T));

  return T;
}
*/

/*
 * Create a tesselation of the 3-sphere (in R4) with at least n cells.
 *
octetramesh_t *hypersphere_tessellation_octetra(int n)
{
  int i;
  octetramesh_t *mesh = init_mesh_S3_octetra();
  octetramesh_t tmp;

  for (i = 0; i < 100000; i++) {

    if (mesh->nt + mesh->no >= n)
      break;

    octetramesh_subdivide(&tmp, mesh);
    printf("\nSubdivision %d...\n", i+1);

    octetramesh_free(mesh);
    free(mesh);
    mesh = octetramesh_clone(&tmp);

    octetramesh_free(&tmp);
  }

  reproject_vertices(mesh->vertices, mesh->nv, mesh->d);
  printf("\nProjection...\n");
  //octetramesh_print_stats(octetramesh_stats(mesh));

  return mesh;
}
*/

/* Multi-resolution tessellation of the 3-sphere based on approximating
 * a scalar function f:S3->R to a given resolution.
 *
octetramesh_t *hypersphere_tessellation_octetra_mres(double(*f)(double *, void *), void *fdata, double resolution)
{
  int i;
  octetramesh_t *mesh = init_mesh_S3_octetra();
  octetramesh_t tmp;

  for (i = 0; i < 100000; i++) {

    octetramesh_subdivide_mres(&tmp, mesh, f, fdata, resolution);

    int nv = mesh->nv;
    int nv2 = tmp.nv;

    if (nv == nv2)  // no subdivision performed
      break;

    printf("\nSubdivision %d;  nv: %d -> %d\n", i+1, nv, nv2);

    octetramesh_free(mesh);
    free(mesh);
    mesh = octetramesh_clone(&tmp);
    octetramesh_free(&tmp);
  }

  reproject_vertices(mesh->vertices, mesh->nv, mesh->d);
  printf("\nProjection...\n");
  octetramesh_print_stats(octetramesh_stats(mesh));

  return mesh;
}
*/

