
#ifndef BINGHAM_UTIL_H
#define BINGHAM_UTIL_H


#define MAXFACT 10000

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))


#define test_alloc(X) do{ if ((void *)(X) == NULL){ fprintf(stderr, "Out of memory in %s, (%s, line %d).\n", __FUNCTION__, __FILE__, __LINE__); exit(1); }} while (0)
#define safe_calloc(x, n, type) do{ x = (type*)calloc(n, sizeof(type)); test_alloc(x); } while (0)
#define safe_malloc(x, n, type) do{ x = (type*)malloc((n)*sizeof(type)); test_alloc(x); } while (0)
#define safe_realloc(x, n, type) do{ x = (type*)realloc(x,(n)*sizeof(type)); test_alloc(x); } while(0)


typedef struct {
  double value;
  void *data;
} sortable_t;

void sort_data(sortable_t *x, size_t n);  /* sort an array of weighted data using qsort */

double get_time_ms();  /* get the current system time in millis */

char *sword(char *s, const char *delim, int n);  /* returns a pointer to the nth word (starting from 0) in string s */



double fact(int x);                                     /* computes the factorial of x */
double lfact(int x);                                    /* computes the log factorial of x */
double surface_area_sphere(int d);                      /* computes the surface area of a unit sphere with dimension d */
double frand();                                         /* returns a random double in [0,1] */
void randperm(int *x, int n, int d);                    /* samples d integers from 0:n-1 uniformly without replacement */
double erfinv(double x);                                /* approximation to the inverse error function */
double normrand(double mu, double sigma);               /* generate a random sample from a normal distribution */

void vnot(int y[], int x[], int n);                                    /* logical not of a binary array */
int count(int x[], int n);                                            /* count the non-zero elements of x */
void find(int *k, int x[], int n);                                    /* computes a dense array of the indices of x's non-zero elements */
void findinv(int *k, int x[], int n);                                 /* computes a sparse array of the indices of x's non-zero elements */
double sum(double x[], int n);                                        /* computes the sum of x's elements */
double max(double x[], int n);                                        /* computes the max of x */
double min(double x[], int n) ;                                       /* computes the min of x */
double norm(double x[], int n);                                       /* computes the norm of x */
double dist(double x[], double y[], int n);                           /* computes the norm of x-y */
double dist2(double x[], double y[], int n);                          /* computes the norm^2 of x-y */
double dot(double x[], double y[], int n);                            /* computes the dot product of z and y */
void add(double z[], double x[], double y[], int n);                  /* adds two vectors, z = x+y */
void sub(double z[], double x[], double y[], int n);                  /* subtracts two vectors, z = x-y */
void mult(double y[], double x[], double c, int n);                   /* multiplies a vector by a scalar, y = c*x */
void normalize(double y[], double x[], int n);                        /* sets y = x/norm(x) */
void vmult(double z[], double x[], double y[], int n);                /* multiplies two vectors, z = x.*y */
void avg(double z[], double x[], double y[], int n);                  /* averages two vectors, z = (x+y)/2 */
void wavg(double z[], double x[], double y[], double w, int n);       /* averages two vectors, z = w*x+(1-w)*y */
void avg3(double y[], double x1[], double x2[], double x3[], int n);  /* averages three vectors, y = (x1+x2+x3)/3 */
void proj(double z[], double x[], double y[], int n);                 /* calculates the projection of x onto y */
int binary_search(double x, double *A, int n);                        /* binary search to find i s.t. A[i-1] <= x < A[i] */

/* transpose a matrix */
void transpose(double **Y, double **X, int n, int m);

/* solve the equation Ax = b, where A is a square n-by-n matrix */
void solve(double x[], double A[], double b[], int n);

/* matrix copy, Y = X */
void matrix_copy(double **Y, double **X, int n, int m);

/* matrix addition, Z = X+Y */
void matrix_add(double **Z, double **X, double **Y, int n, int m);

/* matrix multiplication, Z = X*Y */
void matrix_mult(double **Z, double **X, double **Y, int n, int p, int m);

/* compute the eigenvalues z and eigenvectors V of a real symmetric n-by-n matrix X */
void eigen_symm(double z[], double **V, double **X, int n);



double triangle_area(double x[], double y[], double z[], int n);                  /* calculate the area of a triangle */
double tetrahedron_volume(double x[], double y[], double z[], double w[], int n);   /* calculate the volume of a tetrahedron */

void sample_simplex(double x[], double **S, int n, int d);            /* sample uniformly from a simplex */

double **new_matrix2(int n, int m);                     /* create a new n-by-m 2d matrix of doubles */
int **new_matrix2i(int n, int m);                       /* create a new n-by-m 2d matrix of ints */
void free_matrix2(double **X);                          /* free a 2d matrix of doubles */
void free_matrix2i(int **X);                            /* free a 2d matrix of ints */


typedef struct ilist {
  int x;
  int len;
  struct ilist *next;
} ilist_t;

ilist_t *ilist_add(ilist_t *x, int a);                  /* add an element to a list */
int ilist_contains(ilist_t *x, int a);                  /* check if a list contains an element */
int ilist_find(ilist_t *x, int a);                      /* find the index of an element in a list (or -1 if not found) */
void ilist_free(ilist_t *x);                            /* free a list */


typedef struct {
  int index;
  ilist_t *neighbors;
  int *edges;
} vertex_t;

typedef struct {
  int i;
  int j;
} edge_t;

typedef struct {
  int i;
  int j;
  int k;
} face_t;

typedef struct {
  int nv;
  int ne;
  vertex_t *vertices;
  edge_t *edges;
} graph_t;

void graph_free(graph_t *g);                                /* free a graph */
int graph_find_edge(graph_t *g, int i, int j);              /* find the index of an edge in a graph */
void graph_smooth(double **dst, double **src, graph_t *g, int d, double w);  /* smooth the edges of a graph */


typedef struct {
  int nv;
  int ne;
  int nf;
  int *vertices;
  edge_t *edges;
  face_t *faces;
  int *nvn;                /* # vertex neighbors */
  int *nen;                /* # edge neighbors */
  int **vertex_neighbors;  /* vertex -> {vertices} */
  int **vertex_edges;      /* vertex -> {edges} */
  int **edge_neighbors;    /* edge -> {vertices} */
  int **edge_faces;        /* edge -> {faces} */
  /* internal vars */
  int _vcap;
  int _ecap;
  int _fcap;
  int *_vncap;
  int *_encap;
} meshgraph_t;

meshgraph_t *meshgraph_new(int vcap, int dcap);
int meshgraph_find_edge(meshgraph_t *g, int i, int j);
int meshgraph_find_face(meshgraph_t *g, int i, int j, int k);
int meshgraph_add_edge(meshgraph_t *g, int i, int j);
int meshgraph_add_face(meshgraph_t *g, int i, int j, int k);


typedef struct {
  unsigned char r;
  unsigned char g;
  unsigned char b;
} color_t;

extern const color_t colormap[256];


typedef struct kdtree {
  int i;                   /* index of x (in the original set of points) */
  int d;                   /* length of x */
  double *x;               /* point location */
  double *bbox_min;        /* coordinates of the bottom-left bounding box corner */
  double *bbox_max;        /* coordinates of the top-right bounding box corner */
  int axis;                /* split dimension */
  struct kdtree *left;
  struct kdtree *right;
} kdtree_t;

kdtree_t *kdtree(double **X, int n, int d);
void kdtree_free(kdtree_t *tree);
int kdtree_NN(kdtree_t *tree, double *x);



#endif
