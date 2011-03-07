
#ifndef BINGHAM_HLL_H
#define BINGHAM_HLL_H


#ifdef __cplusplus
extern "C" {
#endif 


  /*
   * Optimizations:
   *  - Fast NN-radius searches
   *  - Precompute local likelihood samples (should store them on disk--probably too slow at runtime)
   */


  typedef struct {
    double **Q;    /* input (S^{dq-1}) */
    double **X;    /* output (R^dx) */
    int n;         /* number of samples (rows) in X,Q */
    int dq;        /* dimension (columns) of Q */
    int dx;        /* dimension (columns) of x */
    double r;      /* kernel radius */
    double *x0;    /* prior mean */
    double **S0;   /* prior covariance */
    double w0;     /* prior weight */
  } hll_t;


  void hll_new(hll_t *hll, double **Q, double **X, int n, int dq, int dx);
  void hll_free(hll_t *hll);
  void hll_sample(double **X, double ***S, double **Q, hll_t *hll, int n);



#ifdef __cplusplus
}
#endif 


#endif

