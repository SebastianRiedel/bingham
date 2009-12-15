
#ifndef BINGHAM_H
#define BINGHAM_H


#ifdef __cplusplus
extern "C" {
#endif 



#include "bingham/tetramesh.h"


typedef struct {
  int d;       /* dimensions */
  double **V;  /* axes */
  double *Z;   /* concentrations */
  double F;    /* normalization constant */
} bingham_t;

typedef struct {
  int n;                /* number of grid cells */
  int d;                /* dimensions */
  double resolution;    /* grid resolution */
  union {               /* cell mesh */
    tetramesh_t *tetramesh;
    /* trimesh_t *trimesh; */
  };
  double **points;      /* cell centroids */
  double *volumes;      /* cell volumes */
  double *mass;         /* cell probability mass */
} bingham_pmf_t;


void bingham_init();
void bingham_new(bingham_t *B, int d, double **V, double *Z);
void bingham_new_S1(bingham_t *B, double *v1, double z1);
void bingham_new_S2(bingham_t *B, double *v1, double *v2, double z1, double z2);
void bingham_new_S3(bingham_t *B, double *v1, double *v2, double *v3, double z1, double z2, double z3);
double bingham_F(bingham_t *B);
double bingham_pdf(double x[], bingham_t *B);
double bingham_L(bingham_t *B, double **X, int n);
void bingham_fit(bingham_t *B, double **X, int n, int d);
void bingham_fit_scatter(bingham_t *B, double **S, int d);
void bingham_sample(double **X, bingham_pmf_t *pmf, int n);
void bingham_discretize(bingham_pmf_t *pmf, bingham_t *B, int ncells);



#ifdef __cplusplus
}
#endif 


#endif

