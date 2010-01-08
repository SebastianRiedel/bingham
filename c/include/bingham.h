
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

typedef struct {
  bingham_t *B;         /* array of binghams */
  double *w;            /* weights */
  int n;                /* number of binghams */
} bingham_mix_t;

void bingham_init();
void bingham_new(bingham_t *B, int d, double **V, double *Z);
void bingham_new_uniform(bingham_t *B, int d);
void bingham_new_S1(bingham_t *B, double *v1, double z1);
void bingham_new_S2(bingham_t *B, double *v1, double *v2, double z1, double z2);
void bingham_new_S3(bingham_t *B, double *v1, double *v2, double *v3, double z1, double z2, double z3);
void bingham_copy(bingham_t *dst, bingham_t *src);
void bingham_alloc(bingham_t *B, int d);
void bingham_free(bingham_t *B);
double bingham_F(bingham_t *B);
double bingham_pdf(double x[], bingham_t *B);
double bingham_L(bingham_t *B, double **X, int n);
void bingham_fit(bingham_t *B, double **X, int n, int d);
void bingham_fit_scatter(bingham_t *B, double **S, int d);
void bingham_discretize(bingham_pmf_t *pmf, bingham_t *B, int ncells);
void bingham_sample(double **X, bingham_pmf_t *pmf, int n);
void bingham_sample_ridge(double **X, bingham_t *B, int n, double pthresh);
void bingham_cluster(bingham_mix_t *BM, double **X, int n, int d);
void bingham_mult(bingham_t *B, bingham_t *B1, bingham_t *B2);
void bingham_mixture_mult(bingham_mix_t *BM, bingham_mix_t *BM1, bingham_mix_t *BM2);
void bingham_mixture_copy(bingham_mix_t *dst, bingham_mix_t *src);
void bingham_mixture_free(bingham_mix_t *BM);
void bingham_mixture_sample_ridge(double **X, bingham_mix_t *BM, int n, double pthresh);
double bingham_mixture_pdf(double x[], bingham_mix_t *BM);
void bingham_mixture_add(bingham_mix_t *dst, bingham_mix_t *src);
double bingham_mixture_peak(bingham_mix_t *BM);
void bingham_mixture_thresh_peaks(bingham_mix_t *BM, double pthresh);
void bingham_mixture_thresh_weights(bingham_mix_t *BM, double wthresh);




#ifdef __cplusplus
}
#endif 


#endif

