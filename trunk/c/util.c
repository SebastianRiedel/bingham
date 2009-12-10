
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include "bingham/util.h"


const color_t colormap[256] =
  {{0, 0, 131},
   {0, 0, 135},
   {0, 0, 139},
   {0, 0, 143},
   {0, 0, 147},
   {0, 0, 151},
   {0, 0, 155},
   {0, 0, 159},
   {0, 0, 163},
   {0, 0, 167},
   {0, 0, 171},
   {0, 0, 175},
   {0, 0, 179},
   {0, 0, 183},
   {0, 0, 187},
   {0, 0, 191},
   {0, 0, 195},
   {0, 0, 199},
   {0, 0, 203},
   {0, 0, 207},
   {0, 0, 211},
   {0, 0, 215},
   {0, 0, 219},
   {0, 0, 223},
   {0, 0, 227},
   {0, 0, 231},
   {0, 0, 235},
   {0, 0, 239},
   {0, 0, 243},
   {0, 0, 247},
   {0, 0, 251},
   {0, 0, 255},
   {0, 4, 255},
   {0, 8, 255},
   {0, 12, 255},
   {0, 16, 255},
   {0, 20, 255},
   {0, 24, 255},
   {0, 28, 255},
   {0, 32, 255},
   {0, 36, 255},
   {0, 40, 255},
   {0, 44, 255},
   {0, 48, 255},
   {0, 52, 255},
   {0, 56, 255},
   {0, 60, 255},
   {0, 64, 255},
   {0, 68, 255},
   {0, 72, 255},
   {0, 76, 255},
   {0, 80, 255},
   {0, 84, 255},
   {0, 88, 255},
   {0, 92, 255},
   {0, 96, 255},
   {0, 100, 255},
   {0, 104, 255},
   {0, 108, 255},
   {0, 112, 255},
   {0, 116, 255},
   {0, 120, 255},
   {0, 124, 255},
   {0, 128, 255},
   {0, 131, 255},
   {0, 135, 255},
   {0, 139, 255},
   {0, 143, 255},
   {0, 147, 255},
   {0, 151, 255},
   {0, 155, 255},
   {0, 159, 255},
   {0, 163, 255},
   {0, 167, 255},
   {0, 171, 255},
   {0, 175, 255},
   {0, 179, 255},
   {0, 183, 255},
   {0, 187, 255},
   {0, 191, 255},
   {0, 195, 255},
   {0, 199, 255},
   {0, 203, 255},
   {0, 207, 255},
   {0, 211, 255},
   {0, 215, 255},
   {0, 219, 255},
   {0, 223, 255},
   {0, 227, 255},
   {0, 231, 255},
   {0, 235, 255},
   {0, 239, 255},
   {0, 243, 255},
   {0, 247, 255},
   {0, 251, 255},
   {0, 255, 255},
   {4, 255, 251},
   {8, 255, 247},
   {12, 255, 243},
   {16, 255, 239},
   {20, 255, 235},
   {24, 255, 231},
   {28, 255, 227},
   {32, 255, 223},
   {36, 255, 219},
   {40, 255, 215},
   {44, 255, 211},
   {48, 255, 207},
   {52, 255, 203},
   {56, 255, 199},
   {60, 255, 195},
   {64, 255, 191},
   {68, 255, 187},
   {72, 255, 183},
   {76, 255, 179},
   {80, 255, 175},
   {84, 255, 171},
   {88, 255, 167},
   {92, 255, 163},
   {96, 255, 159},
   {100, 255, 155},
   {104, 255, 151},
   {108, 255, 147},
   {112, 255, 143},
   {116, 255, 139},
   {120, 255, 135},
   {124, 255, 131},
   {128, 255, 128},
   {131, 255, 124},
   {135, 255, 120},
   {139, 255, 116},
   {143, 255, 112},
   {147, 255, 108},
   {151, 255, 104},
   {155, 255, 100},
   {159, 255, 96},
   {163, 255, 92},
   {167, 255, 88},
   {171, 255, 84},
   {175, 255, 80},
   {179, 255, 76},
   {183, 255, 72},
   {187, 255, 68},
   {191, 255, 64},
   {195, 255, 60},
   {199, 255, 56},
   {203, 255, 52},
   {207, 255, 48},
   {211, 255, 44},
   {215, 255, 40},
   {219, 255, 36},
   {223, 255, 32},
   {227, 255, 28},
   {231, 255, 24},
   {235, 255, 20},
   {239, 255, 16},
   {243, 255, 12},
   {247, 255, 8},
   {251, 255, 4},
   {255, 255, 0},
   {255, 251, 0},
   {255, 247, 0},
   {255, 243, 0},
   {255, 239, 0},
   {255, 235, 0},
   {255, 231, 0},
   {255, 227, 0},
   {255, 223, 0},
   {255, 219, 0},
   {255, 215, 0},
   {255, 211, 0},
   {255, 207, 0},
   {255, 203, 0},
   {255, 199, 0},
   {255, 195, 0},
   {255, 191, 0},
   {255, 187, 0},
   {255, 183, 0},
   {255, 179, 0},
   {255, 175, 0},
   {255, 171, 0},
   {255, 167, 0},
   {255, 163, 0},
   {255, 159, 0},
   {255, 155, 0},
   {255, 151, 0},
   {255, 147, 0},
   {255, 143, 0},
   {255, 139, 0},
   {255, 135, 0},
   {255, 131, 0},
   {255, 128, 0},
   {255, 124, 0},
   {255, 120, 0},
   {255, 116, 0},
   {255, 112, 0},
   {255, 108, 0},
   {255, 104, 0},
   {255, 100, 0},
   {255, 96, 0},
   {255, 92, 0},
   {255, 88, 0},
   {255, 84, 0},
   {255, 80, 0},
   {255, 76, 0},
   {255, 72, 0},
   {255, 68, 0},
   {255, 64, 0},
   {255, 60, 0},
   {255, 56, 0},
   {255, 52, 0},
   {255, 48, 0},
   {255, 44, 0},
   {255, 40, 0},
   {255, 36, 0},
   {255, 32, 0},
   {255, 28, 0},
   {255, 24, 0},
   {255, 20, 0},
   {255, 16, 0},
   {255, 12, 0},
   {255, 8, 0},
   {255, 4, 0},
   {255, 0, 0},
   {251, 0, 0},
   {247, 0, 0},
   {243, 0, 0},
   {239, 0, 0},
   {235, 0, 0},
   {231, 0, 0},
   {227, 0, 0},
   {223, 0, 0},
   {219, 0, 0},
   {215, 0, 0},
   {211, 0, 0},
   {207, 0, 0},
   {203, 0, 0},
   {199, 0, 0},
   {195, 0, 0},
   {191, 0, 0},
   {187, 0, 0},
   {183, 0, 0},
   {179, 0, 0},
   {175, 0, 0},
   {171, 0, 0},
   {167, 0, 0},
   {163, 0, 0},
   {159, 0, 0},
   {155, 0, 0},
   {151, 0, 0},
   {147, 0, 0},
   {143, 0, 0},
   {139, 0, 0},
   {135, 0, 0},
   {131, 0, 0},
   {128, 0, 0}};


double get_time_ms()
{
  struct timeval tv;
  struct timezone tz;

  gettimeofday(&tv, &tz);

  return 1000.*tv.tv_sec + tv.tv_usec/1000.;
}


// computes the log factorial of x
double lfact(int x)
{
  static double logf[MAXFACT];
  static int first = 1;
  int i;

  if (first) {
    first = 0;
    logf[0] = 0;
    for (i = 1; i < MAXFACT; i++)
      logf[i] = log(i) + logf[i-1];
  }

  return logf[x];
}


// computes the factorial of x
double fact(int x)
{
  return exp(lfact(x));
}


// computes the surface area of a unit sphere with dimension d
double surface_area_sphere(int d)
{
  switch(d) {
  case 0:
    return 2;
  case 1:
    return 2*M_PI;
  case 2:
    return 4*M_PI;
  case 3:
    return 2*M_PI*M_PI;
  }

  return (2*M_PI/((double)d-1))*surface_area_sphere(d-2);
}


// count the non-zero elements of x
int count(int x[], int n)
{
  int i;
  int cnt = 0;
  for (i = 0; i < n; i++)
    if (x[i] != 0)
      cnt++;

  return cnt;
}


// returns a dense array of the indices of x's non-zero elements
void find(int *k, int x[], int n)
{
  int i;
  int cnt = 0;
  for (i = 0; i < n; i++)
    if (x[i] != 0)
      k[cnt++] = i;
}


// returns a sparse array of the indices of x's non-zero elements
void findinv(int *k, int x[], int n)
{
  int i;
  int cnt = 0;
  for (i = 0; i < n; i++)
    if (x[i] != 0)
      k[i] = cnt++;
}


// computes the sum of x's elements
double sum(double x[], int n)
{
  int i;
  double y = 0;
  for (i = 0; i < n; i++)
    y += x[i];
  return y;
}


// computes the max of x
double max(double x[], int n)
{
  int i;

  double y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] > y)
      y = x[i];

  return y;
}


// computes the min of x
double min(double x[], int n)
{
  int i;

  double y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] < y)
      y = x[i];

  return y;
}


// computes the norm of x
double norm(double x[], int n)
{
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += x[i]*x[i];

  return sqrt(d);
}

// computes the norm of x-y
double dist(double x[], double y[], int n)
{
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += (x[i]-y[i])*(x[i]-y[i]);

  return sqrt(d);
}


// computes the norm^2 of x-y
double dist2(double x[], double y[], int n)
{
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += (x[i]-y[i])*(x[i]-y[i]);

  return d;
}


// computes the dot product of z and y
double dot(double x[], double y[], int n)
{
  int i;
  double z = 0.0;
  for (i = 0; i < n; i++)
    z += x[i]*y[i];
  return z;
}

// adds two vectors, z = x+y
void add(double z[], double x[], double y[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i] + y[i];
}


// subtracts two vectors, z = x-y
void sub(double z[], double x[], double y[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i] - y[i];
}


// multiplies a vector by a scalar, y = c*x
void mult(double y[], double x[], double c, int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] = c*x[i];
}


// multiplies two vectors, z = x.*y
void vmult(double z[], double x[], double y[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i]*y[i];
}


// averages two vectors, z = (x+y)/2
void avg(double z[], double x[], double y[], int n)
{
  add(z, x, y, n);
  mult(z, z, .5, n);
}


// averages two vectors, z = w*x+(1-w)*y
void wavg(double z[], double x[], double y[], double w, int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = w*x[i] + (1-w)*y[i];
}


// averages three vectors, y = (x1+x2+x3)/3
void avg3(double y[], double x1[], double x2[], double x3[], int n)
{
  add(y, x1, x2, n);
  add(y, y, x3, n);
  mult(y, y, 1/3.0, n);
}


// calculate the projection of x onto y
void proj(double z[], double x[], double y[], int n)
{
  double u[n];  // y's unit vector
  double d = norm(y, n);
  mult(u, y, 1/d, n);
  mult(z, u, dot(x,u,n), n);
}


// add an element to the front of a list
ilist_t *ilist_add(ilist_t *x, int a)
{
  ilist_t *head;
  safe_malloc(head, 1, ilist_t);
  head->x = a;
  head->next = x;
  head->len = (x ? 1 + x->len : 1);

  return head;
}


// check if a list contains an element
int ilist_contains(ilist_t *x, int a)
{
  if (!x)
    return 0;

  ilist_t *tmp;
  for (tmp = x; tmp; tmp = tmp->next)
    if (tmp->x == a)
      return 1;
  return 0;
}


// find the index of an element in a list (or -1 if not found)
int ilist_find(ilist_t *x, int a)
{
  int i = 0;
  ilist_t *tmp;
  for (tmp = x; tmp; tmp = tmp->next) {
    if (tmp->x == a)
      return i;
    i++;
  }

  return -1;
}


// free a list
void ilist_free(ilist_t *x)
{
  ilist_t *tmp, *tmp2;
  tmp = x;
  while (tmp) {
    tmp2 = tmp->next;
    free(tmp);
    tmp = tmp2;
  }  
}


// returns a random double in [0,1]
double frand()
{
  static int first = 1;
  if (first) {
    first = 0;
    srand (time(NULL));
  }

  return fabs(rand()) / (double)RAND_MAX;
}


// approximation to the inverse error function
double erfinv(double x)
{
  if (x < 0)
    return -erfinv(-x);

  double a = .147;

  double y1 = (2/(M_PI*a) + log(1-x*x)/2.0);
  double y2 = sqrt(y1*y1 - (1/a)*log(1-x*x));
  double y3 = sqrt(y2 - y1);
  
  return y3;
}


// generate a random sample from a normal distribution
double normrand(double mu, double sigma)
{
  double u = frand();
  
  return mu + sigma*sqrt(2.0)*erfinv(2*u-1);
}


// create a new n-by-m 2d matrix of doubles
double **new_matrix2(int n, int m)
{
  int i;
  double *raw, **X;
  safe_calloc(raw, n*m, double);
  safe_malloc(X, n, double*);

  for (i = 0; i < n; i++)
    X[i] = raw + m*i;

  return X;
}

// create a new n-by-m 2d matrix of ints
int **new_matrix2i(int n, int m)
{
  int i, *raw, **X;
  safe_calloc(raw, n*m, int);
  safe_malloc(X, n, int*);

  for (i = 0; i < n; i++)
    X[i] = raw + m*i;

  return X;
}

// free a 2d matrix of doubles
void free_matrix2(double **X)
{
  free(X[0]);
  free(X);
}

// free a 2d matrix of ints
void free_matrix2i(int **X)
{
  free(X[0]);
  free(X);
}


// calculate the area of a triangle
double triangle_area(double x[], double y[], double z[], int n)
{
  double a = dist(x, y, n);
  double b = dist(x, z, n);
  double c = dist(y, z, n);
  double s = .5*(a + b + c);

  return sqrt(s*(s-a)*(s-b)*(s-c));
}


// calculate the volume of a tetrahedron
double tetrahedron_volume(double x1[], double x2[], double x3[], double x4[], int n)
{
  double U = dist2(x1, x2, n);
  double V = dist2(x1, x3, n);
  double W = dist2(x2, x3, n);
  double u = dist2(x3, x4, n);
  double v = dist2(x2, x4, n);
  double w = dist2(x1, x4, n);

  double a = v+w-U;
  double b = w+u-V;
  double c = u+v-W;

  return sqrt( (4*u*v*w - u*a*a - v*b*b - w*c*c + a*b*c) / 12.0 );
}


// calculate the volume of a tetrahedron
inline double tetrahedron_volume_old(double x[], double y[], double z[], double w[], int n)
{
  // make an orthonormal basis in the xyz plane (with x at the origin)
  double u[n], v[n], v_proj[n];
  sub(u, y, x, n);             // u = y-x
  sub(v, z, x, n);             // v = z-x
  proj(v_proj, v, u, n);       // project v onto u
  sub(v, v, v_proj, n);        // v -= v_proj
  mult(u, u, 1/norm(u,n), n);  // normalize u
  mult(v, v, 1/norm(v,n), n);  // normalize v

  // project (w-x) onto xyz plane
  double w2[n], wu[n], wv[n], w_proj[n];
  sub(w2, w, x, n);            // w2 = w-x
  proj(wu, w2, u, n);          // project w2 onto u
  proj(wv, w2, v, n);          // project w2 onto v
  add(w_proj, wu, wv, n);      // w_proj = wu + wv
  sub(w2, w2, w_proj, n);      // w2 -= w_proj

  double h = norm(w2, n);  // height
  double A = triangle_area(x, y, z, n);

  return h*A/3.0;
}


// transpose a matrix
void transpose(double **Y, double **X, int n, int m)
{
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      Y[j][i] = X[i][j];
}


// matrix multiplication, Z = X*Y, where X is n-by-p and Y is p-by-m
void matrix_mult(double **Z, double **X, double **Y, int n, int p, int m)
{
  int i, j, k;
  for (i = 0; i < n; i++) {     // row i
    for (j = 0; j < m; j++) {   // column j
      Z[i][j] = 0;
      for (k = 0; k < p; k++)
	Z[i][j] += X[i][k]*Y[k][j];
    }
  }
}


// solve the equation Ax = b, where A is a square n-by-n matrix
void solve(double x[], double A[], double b[], int n)
{
  int s;
  gsl_matrix_view A_gsl = gsl_matrix_view_array(A, n, n);
  gsl_vector_view b_gsl = gsl_vector_view_array(b, n);
  gsl_vector *x_gsl = gsl_vector_alloc(n);
  gsl_permutation *p = gsl_permutation_alloc(n);
     
  gsl_linalg_LU_decomp(&A_gsl.matrix, p, &s);
  gsl_linalg_LU_solve(&A_gsl.matrix, p, &b_gsl.vector, x_gsl);

  memcpy(x, x_gsl->data, n*sizeof(double));

  gsl_permutation_free(p);
  gsl_vector_free(x_gsl);
}


// compute the eigenvalues z and eigenvectors V of a real symmetric n-by-n matrix X
void eigen_symm(double z[], double **V, double **X, int n)
{
  gsl_matrix_view m = gsl_matrix_view_array(X[0], n, n);
  gsl_vector *eval = gsl_vector_alloc(n);
  gsl_matrix *evec = gsl_matrix_alloc(n, n);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(n);

  gsl_eigen_symmv (&m.matrix, eval, evec, w);  
  gsl_eigen_symmv_free(w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

  memcpy(z, eval->data, n*sizeof(double));
  memcpy(V[0], evec->data, n*n*sizeof(double));

  gsl_vector_free(eval);
  gsl_matrix_free(evec);
}


/* create a new graph
graph_t *graph_new(int num_vertices, int edge_capacity)
{
  int i;
  graph_t *g = (graph_t *)malloc(sizeof(graph_t));

  g->nv = num_vertices;
  g->vertices = (vertex_t *)calloc(g->nv, sizeof(vertex_t));
  for (i = 0; i < g->nv; i++)
    g->vertices[i].index = i;

  g->ne = 0;
  g->_edge_capacity = edge_capacity;
  g->edges = (edge_t *)calloc(edge_capacity, sizeof(edge_t));

  return g;
}
*/


// free a graph
void graph_free(graph_t *g)
{
  int i;
  free(g->edges);
  for (i = 0; i < g->nv; i++) {
    free(g->vertices[i].edges);
    ilist_free(g->vertices[i].neighbors);
  }
  free(g);
}


/* add an edge to a graph
void graph_add_edge(graph_t *g, int i, int j)
{
  ilist
  for 


  if (g->ne == g->_edge_capacity) {
    g->_edge_capacity *= 2;
    g->edges = (edge_t *)realloc(g->edges, g->_edge_capacity * sizeof(edge_t));
  }

  g->edges[g->ne].i = i;
  g->edges[g->ne].j = j;
  g->ne++;

  g->vertices[i]
}
*/


// find the index of an edge in a graph
int graph_find_edge(graph_t *g, int i, int j)
{
  int k = ilist_find(g->vertices[i].neighbors, j);

  if (k < 0)
    return -1;

  return g->vertices[i].edges[k];
}


// smooth the edges of a graph
void graph_smooth(double **dst, double **src, graph_t *g, int d, double w)
{
  int i, j;
  double p[d];
  ilist_t *v;

  if (dst != src)
    memcpy(dst[0], src[0], d*g->nv*sizeof(double));

  for (i = 0; i < g->nv; i++) {
    memset(p, 0, d * sizeof(double));                        // p = 0
    for (v = g->vertices[i].neighbors; v; v = v->next) {
      j = v->x;
      add(p, p, dst[j], d);                                  // p += dst[j]
    }
    mult(p, p, 1/norm(p, d), d);                             // p = p/norm(p)

    wavg(dst[i], p, dst[i], w, d);                           // dst[i] = w*p + (1-w)*dst[i]
  }
}


static int dcomp(const void *px, const void *py)
{
  double x = *(double *)px;
  double y = *(double *)py;

  if (x == y)
    return 0;

  return (x < y ? -1 : 1);
}

// sample uniformly from a simplex S with n vertices
void sample_simplex(double x[], double **S, int n, int d)
{
  int i;

  // get n-1 uniform samples, u, on [0,1], and sort them
  double u[n];
  for (i = 0; i < n-1; i++)
    u[i] = frand();
  u[n-1] = 1;

  qsort((void *)u, n-1, sizeof(double), dcomp);

  // mixing coefficients are the order statistics of u
  double c[n];
  c[0] = u[0];
  for (i = 1; i < n; i++)
    c[i] = u[i] - u[i-1];

  // x = sum(c[i]*S[i])
  mult(x, S[0], c[0], d);
  for (i = 1; i < n; i++) {
    double y[d];
    mult(y, S[i], c[i], d);
    add(x, x, y, d);
  }
}


/*******************
typedef struct {
  int nv;
  int ne;
  int nf;
  int *vertices;
  edge_t *edges;
  face_t *faces;
  int *nvn;  // # vertex neighbors
  int *nen;  // # edge neighbors
  int **vertex_neighbors;  // vertex -> {vertices}
  int **vertex_edges;      // vertex -> {edges}
  int **edge_neighbors;    // edge -> {vertices}
  int **edge_faces;        // edge -> {faces}
  // internal vars
  int _vcap;
  int _ecap;
  int _fcap;
  int *_vncap;
  int *_encap;
} meshgraph_t;
******************/


/*
 * Create a new meshgraph with initial vertex capacity 'vcap' and degree capacity 'dcap'.
 */
meshgraph_t *meshgraph_new(int vcap, int dcap)
{
  int i;
  meshgraph_t *g;
  safe_calloc(g, 1, meshgraph_t);

  safe_malloc(g->vertices, vcap, int);
  safe_malloc(g->edges, vcap, edge_t);
  safe_malloc(g->faces, vcap, face_t);

  g->_vcap = g->_ecap = g->_fcap = vcap;
  safe_malloc(g->_vncap, vcap, int);
  safe_malloc(g->_encap, vcap, int);

  safe_malloc(g->vertex_neighbors, vcap, int *);
  safe_malloc(g->vertex_edges, vcap, int *);
  safe_calloc(g->nvn, vcap, int);
  for (i = 0; i < vcap; i++) {
    safe_malloc(g->vertex_neighbors[i], dcap, int);
    safe_malloc(g->vertex_edges[i], dcap, int);
    g->_vncap[i] = dcap;
  }

  safe_malloc(g->edge_neighbors, vcap, int *);
  safe_malloc(g->edge_faces, vcap, int *);
  safe_calloc(g->nen, vcap, int);
  for (i = 0; i < vcap; i++) {
    safe_malloc(g->edge_neighbors[i], dcap, int);
    safe_malloc(g->edge_faces[i], dcap, int);
    g->_encap[i] = dcap;
  }

  return g;
}


void meshgraph_free(meshgraph_t *g)
{
  int i;

  free(g->vertices);
  free(g->edges);
  free(g->faces);
  free(g->nvn);
  free(g->nen);
  free(g->_vncap);
  free(g->_encap);

  for (i = 0; i < g->nv; i++) {
    free(g->vertex_neighbors[i]);
    free(g->vertex_edges[i]);
  }
  free(g->vertex_neighbors);
  free(g->vertex_edges);

  for (i = 0; i < g->ne; i++) {
    free(g->edge_neighbors[i]);
    free(g->edge_faces[i]);
  }
  free(g->edge_neighbors);
  free(g->edge_faces);

  free(g);
}


int meshgraph_find_edge(meshgraph_t *g, int i, int j)
{
  int n;
  for (n = 0; n < g->nvn[i]; n++)
    if (g->vertex_neighbors[i][n] == j)
      return g->vertex_edges[i][n];

  return -1;
}


int meshgraph_find_face(meshgraph_t *g, int i, int j, int k)
{
  int e = meshgraph_find_edge(g, i, j);
  if (e < 0)
    return -1;

  int n;
  for (n = 0; n < g->nen[e]; n++)
    if (g->edge_neighbors[e][n] == k)
      return g->edge_faces[e][n];

  return -1;
}


static inline int meshgraph_add_vertex_neighbor(meshgraph_t *g, int i, int vertex, int edge)
{
  int n = g->nvn[i];
  if (n == g->_vncap[i]) {
    g->_vncap[i] *= 2;
    safe_realloc(g->vertex_neighbors[i], g->_vncap[i], int);
    safe_realloc(g->vertex_edges[i], g->_vncap[i], int);
  }
  g->vertex_neighbors[i][n] = vertex;
  g->vertex_edges[i][n] = edge;
  g->nvn[i]++;

  return n;
}


int meshgraph_add_edge(meshgraph_t *g, int i, int j)
{
  //printf("meshgraph_add_edge(%d, %d)\n", i, j);

  int edge = meshgraph_find_edge(g, i, j);
  if (edge >= 0)
    return edge;

  //printf("  break 1\n");

  // add the edge
  if (g->ne == g->_ecap) {
    int old_ecap = g->_ecap;
    g->_ecap *= 2;
    safe_realloc(g->edges, g->_ecap, edge_t);
    safe_realloc(g->edge_neighbors, g->_ecap, int *);
    safe_realloc(g->edge_faces, g->_ecap, int *);
    safe_realloc(g->nen, g->_ecap, int);
    safe_realloc(g->_encap, g->_ecap, int);

    //printf("  break 1.1\n");

    int e;
    for (e = old_ecap; e < g->_ecap; e++) {
      //printf("    e = %d\n", e);
      g->nen[e] = 0;
      int dcap = g->_encap[0];

      //printf("    dcap = %d\n", dcap);

      safe_malloc(g->edge_neighbors[e], dcap, int);

      //printf("    break 1.1.1\n");

      safe_malloc(g->edge_faces[e], dcap, int);

      //printf("    break 1.1.2\n");

      g->_encap[e] = dcap;
    }
  }

  //printf("  break 2\n");

  edge = g->ne;
  g->edges[edge].i = i;
  g->edges[edge].j = j;
  g->ne++;

  //printf("  break 3\n");

  // add the vertex neighbors
  meshgraph_add_vertex_neighbor(g, i, j, edge);
  meshgraph_add_vertex_neighbor(g, j, i, edge);

  //printf("  break 4\n");

  return edge;
}


static inline int meshgraph_add_edge_neighbor(meshgraph_t *g, int i, int vertex, int face)
{
  int n = g->nen[i];

  //printf("g->nen[%d] = %d, g->_encap[%d] = %d\n", i, n, i, g->_encap[i]);

  if (n == g->_encap[i]) {

    //printf("n == g->_encap[%d]\n", i);

    g->_encap[i] *= 2;
    safe_realloc(g->edge_neighbors[i], g->_encap[i], int);
    safe_realloc(g->edge_faces[i], g->_encap[i], int);
  }

  //printf("  break 1\n");

  g->edge_neighbors[i][n] = vertex;

  //printf("  break 2\n");

  g->edge_faces[i][n] = face;

  //printf("  break 3\n");

  g->nen[i]++;

  return n;
}


int meshgraph_add_face(meshgraph_t *g, int i, int j, int k)
{
  //printf("meshgraph_add_face(%d, %d, %d)\n", i, j, k);

  int face = meshgraph_find_face(g, i, j, k);
  if (face >= 0)
    return face;

  //printf("  break 1\n");

  // add the edges
  int edge_ij = meshgraph_add_edge(g, i, j);
  int edge_ik = meshgraph_add_edge(g, i, k);
  int edge_jk = meshgraph_add_edge(g, j, k);

  //printf("  break 2\n");

  // add the face
  //printf("g->nf = %d, g->_fcap = %d\n", g->nf, g->_fcap);

  if (g->nf == g->_fcap) {
    g->_fcap *= 2;
    safe_realloc(g->faces, g->_fcap, face_t);
  }
  face = g->nf;
  g->faces[face].i = i;
  g->faces[face].j = j;
  g->faces[face].k = k;
  g->nf++;

  //printf("  break 3\n");

  // add the edge neighbors
  meshgraph_add_edge_neighbor(g, edge_ij, k, face);
  meshgraph_add_edge_neighbor(g, edge_ik, j, face);
  meshgraph_add_edge_neighbor(g, edge_jk, i, face);

  //printf("  break 4\n");

  return face;
}




int qselect(double *x, int n, int k)
{
  if (n == 1)
    return 0;

  double pivot = x[k];

  // partition x into y < pivot, z > pivot
  int i, ny=0, nz=0;
  for (i = 0; i < n; i++) {
    if (x[i] < pivot)
      ny++;
    else if (x[i] > pivot)
      nz++;
  }

  if (k < ny) {
    double *y;
    int *yi;
    safe_calloc(y, ny, double);
    safe_calloc(yi, ny, int);
    ny = 0;
    for (i = 0; i < n; i++) {
      if (x[i] < pivot) {
	yi[ny] = i;
	y[ny++] = x[i];
      }
    }
    i = yi[qselect(y, ny, k)];
    free(y);
    free(yi);
    return i;
  }

  else if (k >= n - nz) {
    double *z;
    int *zi;
    safe_calloc(z, nz, double);
    safe_calloc(zi, nz, int);
    nz = 0;
    for (i = 0; i < n; i++) {
      if (x[i] > pivot) {
	zi[nz] = i;
	z[nz++] = x[i];
      }
    }
    i = zi[qselect(z, nz, k-(n-nz))];
    free(z);
    free(zi);
    return i;
  }

  return k;
}


static kdtree_t *build_kdtree(double **X, int *xi, int n, int d, int depth)
{
  if (n == 0)
    return NULL;

  int i, axis = depth % d;
  kdtree_t *node;
  safe_calloc(node, 1, kdtree_t);
  node->axis = axis;

  double *x;
  safe_calloc(x, n, double);
  for (i = 0; i < n; i++)
    x[i] = X[i][axis];

  int median = qselect(x, n, n/2);

  // node location
  node->i = xi[median];
  node->d = d;
  safe_malloc(node->x, d, double);
  memcpy(node->x, X[median], d*sizeof(double));

  // node bbox init:  bbox_min = bbox_max = x
  safe_malloc(node->bbox_min, d, double);
  safe_malloc(node->bbox_max, d, double);
  memcpy(node->bbox_min, node->x, d*sizeof(double));
  memcpy(node->bbox_max, node->x, d*sizeof(double));

  // partition x into y < pivot, z > pivot
  double pivot = x[median];
  int ny=0, nz=0;
  for (i = 0; i < n; i++) {
    if (i == median)
      continue;
    if (x[i] <= pivot)
      ny++;
    else if (x[i] > pivot)
      nz++;
  }

  //printf("n = %d, d = %d, depth = %d, axis = %d --> median = %d, X[median] = (%f, %f, %f), ny = %d, nz = %d\n",
  //	 n, d, depth, axis, median, X[median][0], X[median][1], X[median][2], ny, nz);


  if (ny > 0) {
    double **Y = new_matrix2(ny, d);
    int *yi;
    safe_calloc(yi, ny, int);
    ny = 0;
    for (i = 0; i < n; i++) {
      if (i == median)
	continue;
      if (x[i] <= pivot) {
	yi[ny] = xi[i];
	memcpy(Y[ny], X[i], d*sizeof(double));
	ny++;
      }
    }
    node->left = build_kdtree(Y, yi, ny, d, depth+1);

    // update bbox
    for (i = 0; i < d; i++) {
      if (node->left->bbox_min[i] < node->bbox_min[i])
	node->bbox_min[i] = node->left->bbox_min[i];
      if (node->left->bbox_max[i] > node->bbox_max[i])
	node->bbox_max[i] = node->left->bbox_max[i];
    }

    free_matrix2(Y);
    free(yi);
  }

  if (nz > 0) {
    double **Z = new_matrix2(nz, d);
    int *zi;
    safe_calloc(zi, nz, int);
    nz = 0;
    for (i = 0; i < n; i++) {
      if (i == median)
	continue;
      if (x[i] > pivot) {
	zi[nz] = xi[i];
	memcpy(Z[nz], X[i], d*sizeof(double));
	nz++;
      }
    }
    node->right = build_kdtree(Z, zi, nz, d, depth+1);

    // update bbox
    for (i = 0; i < d; i++) {
      if (node->right->bbox_min[i] < node->bbox_min[i])
	node->bbox_min[i] = node->right->bbox_min[i];
      if (node->right->bbox_max[i] > node->bbox_max[i])
	node->bbox_max[i] = node->right->bbox_max[i];
    }

    free_matrix2(Z);
    free(zi);
  }

  free(x);
  return node;
}


kdtree_t *kdtree(double **X, int n, int d)
{
  int i, *xi;
  safe_malloc(xi, n, int);
  for (i = 0; i < n; i++)
    xi[i] = i;

  kdtree_t *tree = build_kdtree(X, xi, n, d, 0);

  free(xi);
  return tree;
}


static kdtree_t *kdtree_NN_node(kdtree_t *tree, double *x, kdtree_t *best)
{
  if (tree == NULL)
    return best;

  //printf("node %d", tree->i);

  int i, d = tree->d;
  double dbest = (best ? dist(x, best->x, d) : DBL_MAX);

  // first, check if any node in tree can possibly be better than 'best'
  if (best) {
    double y[d];  // closest point on the tree's bbox to x
    for (i = 0; i < d; i++) {
      if (x[i] < tree->bbox_min[i])
	y[i] = tree->bbox_min[i];
      else if (x[i] > tree->bbox_max[i])
	y[i] = tree->bbox_max[i];
      else
	y[i] = x[i];
    }
    if (dist(y, x, d) >= dbest) {  // 'best' is closer than the closest possible point in tree, so return
      //printf("  --> pruned!\n");
      return best;
    }
  }

  int axis = tree->axis;
  kdtree_t *nn = best;

  // compare with the node itself
  double dtree = dist(x, tree->x, d);
  //printf(" (%f)", dtree);
  if (dtree < dbest) {
    nn = tree;
    dbest = dtree;
    //printf(" --> new best");
  }
  //printf("\n");

  // compare with the NN in each sub-tree
  if (x[axis] <= tree->x[axis]) {
    nn = kdtree_NN_node(tree->left, x, nn);
    nn = kdtree_NN_node(tree->right, x, nn);
  }
  else if (x[axis] > tree->x[axis]) {
    nn = kdtree_NN_node(tree->right, x, nn);
    nn = kdtree_NN_node(tree->left, x, nn);
  }

  //dbest = dist(x, nn->x, d);
  //printf(" ... return %d (%f)\n", nn->i, dbest);

  return nn;
}

int kdtree_NN(kdtree_t *tree, double *x)
{
  kdtree_t *nn = kdtree_NN_node(tree, x, NULL);
  return (nn ? nn->i : -1);
}


void kdtree_free(kdtree_t *tree)
{
  if (tree == NULL)
    return;

  kdtree_free(tree->left);
  kdtree_free(tree->right);

  free(tree->x);
  free(tree->bbox_min);
  free(tree->bbox_max);

  free(tree);
}


