
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/bingham_constants.h"
#include "bingham/hypersphere.h"





//---------------------  main  ---------------------//



void usage(char *argv[])
{
  printf("usage: %s <2d|3d> <z_min> <z_max> <step>\n", argv[0]);
  exit(1);
}



void test_fit_quaternions(int argc, char *argv[])
{
  int n = 101;
  double X[101][4] = {
    {0.8776,         0,         0,   -0.4794},
    {0.8752,         0,         0,   -0.4838},
    {0.8727,         0,         0,   -0.4882},
    {0.8703,         0,         0,   -0.4925},
    {0.8678,         0,         0,   -0.4969},
    {0.8653,         0,         0,   -0.5012},
    {0.8628,         0,         0,   -0.5055},
    {0.8603,         0,         0,   -0.5098},
    {0.8577,         0,         0,   -0.5141},
    {0.8551,         0,         0,   -0.5184},
    {0.8525,         0,         0,   -0.5227},
    {0.8499,         0,         0,   -0.5269},
    {0.8473,         0,         0,   -0.5312},
    {0.8446,         0,         0,   -0.5354},
    {0.8419,         0,         0,   -0.5396},
    {0.8392,         0,         0,   -0.5438},
    {0.8365,         0,         0,   -0.5480},
    {0.8337,         0,         0,   -0.5522},
    {0.8309,         0,         0,   -0.5564},
    {0.8281,         0,         0,   -0.5605},
    {0.8253,         0,         0,   -0.5646},
    {0.8225,         0,         0,   -0.5688},
    {0.8196,         0,         0,   -0.5729},
    {0.8168,         0,         0,   -0.5770},
    {0.8139,         0,         0,   -0.5810},
    {0.8110,         0,         0,   -0.5851},
    {0.8080,         0,         0,   -0.5891},
    {0.8051,         0,         0,   -0.5932},
    {0.8021,         0,         0,   -0.5972},
    {0.7991,         0,         0,   -0.6012},
    {0.7961,         0,         0,   -0.6052},
    {0.7930,         0,         0,   -0.6092},
    {0.7900,         0,         0,   -0.6131},
    {0.7869,         0,         0,   -0.6171},
    {0.7838,         0,         0,   -0.6210},
    {0.7807,         0,         0,   -0.6249},
    {0.7776,         0,         0,   -0.6288},
    {0.7744,         0,         0,   -0.6327},
    {0.7712,         0,         0,   -0.6365},
    {0.7681,         0,         0,   -0.6404},
    {0.7648,         0,         0,   -0.6442},
    {0.7616,         0,         0,   -0.6480},
    {0.7584,         0,         0,   -0.6518},
    {0.7551,         0,         0,   -0.6556},
    {0.7518,         0,         0,   -0.6594},
    {0.7485,         0,         0,   -0.6631},
    {0.7452,         0,         0,   -0.6669},
    {0.7418,         0,         0,   -0.6706},
    {0.7385,         0,         0,   -0.6743},
    {0.7351,         0,         0,   -0.6780},
    {0.7317,         0,         0,   -0.6816},
    {0.7283,         0,         0,   -0.6853},
    {0.7248,         0,         0,   -0.6889},
    {0.7214,         0,         0,   -0.6925},
    {0.7179,         0,         0,   -0.6961},
    {0.7144,         0,         0,   -0.6997},
    {0.7109,         0,         0,   -0.7033},
    {0.7074,         0,         0,   -0.7068},
    {0.7038,         0,         0,   -0.7104},
    {0.7003,         0,         0,   -0.7139},
    {0.6967,         0,         0,   -0.7174},
    {0.6931,         0,         0,   -0.7208},
    {0.6895,         0,         0,   -0.7243},
    {0.6859,         0,         0,   -0.7277},
    {0.6822,         0,         0,   -0.7311},
    {0.6786,         0,         0,   -0.7345},
    {0.6749,         0,         0,   -0.7379},
    {0.6712,         0,         0,   -0.7413},
    {0.6675,         0,         0,   -0.7446},
    {0.6637,         0,         0,   -0.7480},
    {0.6600,         0,         0,   -0.7513},
    {0.6562,         0,         0,   -0.7546},
    {0.6524,         0,         0,   -0.7578},
    {0.6486,         0,         0,   -0.7611},
    {0.6448,         0,         0,   -0.7643},
    {0.6410,         0,         0,   -0.7675},
    {0.6372,         0,         0,   -0.7707},
    {0.6333,         0,         0,   -0.7739},
    {0.6294,         0,         0,   -0.7771},
    {0.6255,         0,         0,   -0.7802},
    {0.6216,         0,         0,   -0.7833},
    {0.6177,         0,         0,   -0.7864},
    {0.6137,         0,         0,   -0.7895},
    {0.6098,         0,         0,   -0.7926},
    {0.6058,         0,         0,   -0.7956},
    {0.6018,         0,         0,   -0.7986},
    {0.5978,         0,         0,   -0.8016},
    {0.5938,         0,         0,   -0.8046},
    {0.5898,         0,         0,   -0.8076},
    {0.5857,         0,         0,   -0.8105},
    {0.5817,         0,         0,   -0.8134},
    {0.5776,         0,         0,   -0.8163},
    {0.5735,         0,         0,   -0.8192},
    {0.5694,         0,         0,   -0.8220},
    {0.5653,         0,         0,   -0.8249},
    {0.5612,         0,         0,   -0.8277},
    {0.5570,         0,         0,   -0.8305},
    {0.5529,         0,         0,   -0.8333},
    {0.5487,         0,         0,   -0.8360},
    {0.5445,         0,         0,   -0.8388},
    {0.5403,         0,         0,   -0.8415}};

  int i, d = 4;

  double *Xp[n];
  for (i = 0; i < n; i++)
    Xp[i] = &X[i][0];

  bingham_t B;
  bingham_fit(&B, Xp, n, d);

  printf("B->Z = [%f %f %f]\n", B.Z[0], B.Z[1], B.Z[2]);
}



void test_fit_simple(int argc, char *argv[])
{
  double X[5][4] = {{1,0,0,0},
		    {1, .1, .4, .1},
		    {1, -.1, .5, .1},
		    {1, .09, .6, 0},
		    {1, .11, -.5, 0}};

  int i, j, n = 5, d = 4;
  double dx;

  // normalize samples X
  for (i = 0; i < n; i++) {
    dx = norm(X[i], d);
    for (j = 0; j < d; j++)
      X[i][j] = X[i][j]/dx;
  }

  double *Xp[5] = {&X[0][0], &X[1][0], &X[2][0], &X[3][0], &X[4][0]};

  bingham_t B;
  bingham_fit(&B, Xp, n, d);
}


void test_fit(int argc, char *argv[])
{
  if (argc < 6) {
    printf("usage: %s <n> <s1> <s2> <s3> <s4>\n", argv[0]);
    return;
  }

  int i, n = atoi(argv[1]);
  double s1 = atof(argv[2]);
  double s2 = atof(argv[3]);
  double s3 = atof(argv[4]);
  double s4 = atof(argv[5]);

  double dx, **X = new_matrix2(n, 4);
  for (i = 0; i < n; i++) {
    X[i][0] = normrand(0, s1);
    X[i][1] = normrand(0, s2);
    X[i][2] = normrand(0, s3);
    X[i][3] = normrand(0, s4);
    dx = norm(X[i], 4);
    mult(X[i], X[i], 1/dx, 4);
  }

  bingham_t B;
  bingham_fit(&B, X, n, 4);

  free_matrix2(X);
}



/*
void test_sample_2d(int argc, char *argv[])
{
  if (argc < 4) {
    printf("usage: %s <z1> <z2> <n>\n", argv[0]);
    return;
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);  
  int i, n = atoi(argv[3]);
  double X[n][3];
  double Z[2] = {z1, z2};
  double V[3][3] = {{0,1,0}, {0,0,1}, {1,0,0}};

  double sigma1 = 1/sqrt(-2*z1);  //sqrt(bingham_dF1_2d(Z[0], Z[1]) / bingham_F_2d(Z[0], Z[1]));
  double sigma2 = 1/sqrt(-2*z2);  //sqrt(bingham_dF2_2d(Z[0], Z[1]) / bingham_F_2d(Z[0], Z[1]));

  printf("X = [ ...\n");
  for (i = 0; i < n; i++) {
    bingham_sample_2d(X[i], Z, V, sigma1, sigma2);
    //uniform_sample_2d(X[i]);
    printf("%f, %f, %f ; ...\n", X[i][0], X[i][1], X[i][2]);
  }
  printf("];\n\n");

  //Z[0] = Z[1] = -5;
  bingham_MLE_2d_grad_desc(Z, V, X, n);
}
*/

/*
void test_sample_3d(int argc, char *argv[])
{
  if (argc < 5) {
    printf("usage: %s <z1> <z2> <z3> <n>\n", argv[0]);
    return;
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);
  int i, n = atoi(argv[4]);
  double X[n][4];
  double Z[3] = {z1, z2, z3};
  double V[4][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}, {1,0,0,0}};

  printf("X = [ ...\n");
  for (i = 0; i < n; i++) {
    bingham_sample_3d(X[i], Z, V);
    printf("%f, %f, %f, %f ; ...\n", X[i][0], X[i][1], X[i][2], X[i][3]);
  }
  printf("];\n\n");

  //Z[0] = Z[1] = -5;
  bingham_MLE_3d_grad_desc(Z, V, X, n);
}
*/

void test_bingham(int argc, char *argv[])
{
  double z_min, z_max, step;
  int dim;

  if (argc < 5)
    usage(argv);

  if (strcmp(argv[1], "2d") == 0)
    dim = 2;
  else if (strcmp(argv[1], "3d") == 0)
    dim = 3;
  else
    usage(argv);

  z_min = atof(argv[2]);
  z_max = atof(argv[3]);
  step = atof(argv[4]);

  printf("\n");
  printf("[X Y] = meshgrid(%f:%f:%f);\n\n", z_min, step, z_max);

  if (dim == 2) {
    compute_all_bingham_F_2d(z_min, z_max, step, z_min, z_max, step);
    compute_all_bingham_dF1_2d(z_min, z_max, step, z_min, z_max, step);
    compute_all_bingham_dF2_2d(z_min, z_max, step, z_min, z_max, step);
  }
  else if (dim == 3) {
    compute_all_bingham_F_3d(z_min, z_max, step, z_min, z_max, step, z_min, z_max, step);
    compute_all_bingham_dF1_3d(z_min, z_max, step, z_min, z_max, step, z_min, z_max, step);
    compute_all_bingham_dF2_3d(z_min, z_max, step, z_min, z_max, step, z_min, z_max, step);
    compute_all_bingham_dF3_3d(z_min, z_max, step, z_min, z_max, step, z_min, z_max, step);
  }
}



// y = -sqrt(z)
static double y_range[] = {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
			   1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00,
			   2.20, 2.40, 2.60, 2.80, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00,
			   4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00,
			   9.50, 10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00, 13.50,
			   14.00, 14.50, 15.00, 15.50, 16.00, 16.50, 17.00, 17.50,
			   18.00, 18.50, 19.00, 19.50, 20.00};
static int y_cnt = sizeof(y_range)/sizeof(y_range[0]);


void compute_bingham_constants(int argc, char *argv[])
{
  int i, dim, k0 = 0, k1 = y_cnt;

  if (argc < 2) {
    printf("usage: %s <2d|3d> [k0] [k1]\n", argv[0]);
    exit(1);
  }

  if (strcmp(argv[1], "2d") == 0)
    dim = 2;
  else if (strcmp(argv[1], "3d") == 0)
    dim = 3;
  else {
    printf("usage: %s <2d|3d> [k0] [k1]\n", argv[0]);
    exit(1);
  }

  if (argc > 2)
    k0 = atoi(argv[2]);

  if (argc > 3)
    k1 = atoi(argv[3]);

  printf("\n");
  printf("y_range = [ ");
  for (i = 0; i < y_cnt; i++)
    printf("%.2f ", y_range[i]);
  printf("];\n\n");

  if (dim == 2) {
    fprintf(stderr, "Computing 2D constants:\n");
    compute_range_bingham_F_2d(y_range, y_cnt);
    compute_range_bingham_dF1_2d(y_range, y_cnt);
    compute_range_bingham_dF2_2d(y_range, y_cnt);
  }
  else if (dim == 3) {
    fprintf(stderr, "Computing 3D constants from k=%d..%d:\n", k0, k1-1);
    compute_range_bingham_F_3d(y_range, y_cnt, k0, k1);
    compute_range_bingham_dF1_3d(y_range, y_cnt, k0, k1);
    compute_range_bingham_dF2_3d(y_range, y_cnt, k0, k1);
    compute_range_bingham_dF3_3d(y_range, y_cnt, k0, k1);
  }
}


void test_bingham_discretize(int argc, char *argv[])
{
  if (argc < 5) {
    printf("usage: %s <z1> <z2> <z3> <ncells>\n", argv[0]);
    exit(1);
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);
  int ncells = atoi(argv[4]);

  double Z[3] = {z1, z2, z3};
  double V[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

  bingham_t B;
  bingham_new(&B, 4, Vp, Z);

  bingham_pmf_t pmf;
  bingham_discretize(&pmf, &B, ncells);

  // check if pmf sums to 1
  double tot_mass = sum(pmf.mass, pmf.n);
  //printf("tot_mass = %f\n", tot_mass);

  int i;

  //printf("break 1\n");
  int *colors; safe_malloc(colors, pmf.n, int);
  //printf("break 2\n");

  double max_mass = max(pmf.mass, pmf.n);
  for (i = 0; i < pmf.n; i++)
    colors[i] = (int)(255*(pmf.mass[i] / max_mass));

  //printf("Calling tetramesh_meshgraph()...");
  double t = get_time_ms();
  meshgraph_t *graph = tetramesh_meshgraph(pmf.tetramesh);
  //printf("%f ms\n", get_time_ms() - t);

  //printf("Calling tetramesh_graph()...");
  t = get_time_ms();
  tetramesh_graph(pmf.tetramesh);
  //printf("%f ms\n", get_time_ms() - t);

  //printf("Calling tetramesh_save_PLY_colors()...\n");

  tetramesh_save_PLY_colors(pmf.tetramesh, graph, "mesh.ply", colors);

  // print out the points
  printf("pmf.points = [ ");
  for (i = 0; i < pmf.n; i++)
    printf("%f %f %f %f ; ", pmf.points[i][0], pmf.points[i][1], pmf.points[i][2], pmf.points[i][3]);
  printf("];\n");
}

/*
void test_bingham_mres(int argc, char *argv[])
{
  if (argc < 5) {
    printf("usage: %s <z1> <z2> <z3> <resolution>\n", argv[0]);
    exit(1);
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);
  double resolution = atof(argv[4]);

  double Z[3] = {z1, z2, z3};
  double V[3][4] = {{0,0,0,1}, {0,1,0,0}, {0,0,1,0}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

  bingham_t B;
  bingham_new(&B, 4, Vp, Z);

  bingham_pmf_t pmf;
  bingham_discretize_mres(&pmf, &B, resolution);

  // check if pmf sums to 1
  double tot_mass = sum(pmf.mass, pmf.n);
  printf("tot_mass = %f\n", tot_mass);

  int i;
  int colors[pmf.n];
  double max_mass = max(pmf.mass, pmf.n);
  for (i = 0; i < pmf.n; i++)
    colors[i] = (int)(255*(pmf.mass[i] / max_mass));

  meshgraph_t *graph = tetramesh_meshgraph(pmf.tetramesh);

  tetramesh_save_PLY_colors(pmf.tetramesh, graph, "mres.ply", colors);
}
*/


void test_bingham_pdf(int argc, char *argv[])
{
  if (argc < 4) {
    printf("usage: %s <z1> <z2> <z3>\n", argv[0]);
    exit(1);
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);

  double Z[3] = {z1, z2, z3};
  double V[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

  bingham_t B;
  bingham_new(&B, 4, Vp, Z);

  double X[5][4] = {{0,0,0,1},
		    {0, .1, .4, 1},
		    {0, -.1, .5, 1},
		    {0, .09, .6, 1},
		    {0, .11, -.5, 1}};

  int i, j, n = 5, d = 4;
  double dx;

  // normalize samples X
  for (i = 0; i < n; i++) {
    dx = norm(X[i], d);
    for (j = 0; j < d; j++)
      X[i][j] = X[i][j]/dx;
  }

  double *Xp[5] = {&X[0][0], &X[1][0], &X[2][0], &X[3][0], &X[4][0]};

  for (i = 0; i < n; i++)
    printf("bingham_pdf(%.2f, %.2f, %.2f, %.2f) = %f\n", X[i][0], X[i][1], X[i][2], X[i][3], bingham_pdf(Xp[i], &B));
}


void test_bingham_sample(int argc, char *argv[])
{
  if (argc < 6) {
    printf("usage: %s <z1> <z2> <z3> <num_cells> <num_samples>\n", argv[0]);
    exit(1);
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);
  int ncells = atoi(argv[4]);
  int nsamples = atoi(argv[5]);

  double Z[3] = {z1, z2, z3};
  double V[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

  bingham_t B;
  bingham_new(&B, 4, Vp, Z);

  bingham_pmf_t pmf;
  double t0 = get_time_ms();
  bingham_discretize(&pmf, &B, ncells);
  printf("Created PMF with %d cells in %.0f ms\n", pmf.n, get_time_ms() - t0);

  t0 = get_time_ms();
  double **X = new_matrix2(nsamples, 4);
  bingham_sample(X, &pmf, nsamples);
  printf("Sampled %d points in %.0f ms\n", nsamples, get_time_ms() - t0);

  bingham_fit(&B, X, nsamples, 4);

  
}

void test_bingham_init()
{
  double t0 = get_time_ms();

  bingham_init();

  double t1 = get_time_ms();

  //printf("Initialized bingham library in %.0f ms\n", t1-t0);
}


int main(int argc, char *argv[])
{
  test_bingham_init();

  //test_fit_quaternions(argc, argv);

  //test_bingham_sample(argc, argv);
  test_bingham_discretize(argc, argv);
  //test_bingham(argc, argv);
  //compute_bingham_constants(argc, argv);
  //test_bingham_pdf(argc, argv);
  //test_fit(argc, argv);


  return 0;
}
