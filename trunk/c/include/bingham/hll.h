
#ifndef BINGHAM_HLL_H
#define BINGHAM_HLL_H


#ifdef __cplusplus
extern "C" {
#endif 


  /*
   * Optimizations:
   *  - For S^3, add to hypersphere.h:
   *     1) a kdtree for fast NN lookup
   *     2) a graph of the tetramesh vertices for fast NN-radius searches via graph traversal
   *     3) functions hypersphere_nn() and hypersphere_nn_radius()
   *
   *  - Precompute local likelihood samples (should store them on disk--probably too slow at runtime)
   */


  typedef struct {
    double **Q;    /* input (S^{dq-1}) */
    double **X;    /* output (R^dx) */
    int n;         /* number of rows */
    int dq;        /* dimension (columns) of Q */
    int dx;        /* dimension (columns) of x */
    double *x0;    /* prior mean */
    double **s0;   /* prior covariance */
    double w0;     /* prior weight */
  } hll_t;


  void hll_new(hll_t *hll, double **Q, double **X, int n, int dq, int dx);
  void hll_sample(double **X, double **Q, hll_t *hll, int n);



#ifdef __cplusplus
}
#endif 


#endif

