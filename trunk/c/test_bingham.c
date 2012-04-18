
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
static double y_range[] = { 0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80,  0.90,  1.00,
			    1.10,  1.20,  1.30,  1.40,  1.50,  1.60,  1.70,  1.80,  1.90,  2.00,
			    2.20,  2.40,  2.60,  2.80,  3.00,  3.20,  3.40,  3.60,  3.80,  4.00,
			    4.50,  5.00,  5.50,  6.00,  6.50,  7.00,  7.50,  8.00,  8.50,  9.00,
			    9.50, 10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00, 13.50, 14.00,
			   14.50, 15.00, 15.50, 16.00, 16.50, 17.00, 17.50, 18.00, 18.50, 19.00,
			   19.50, 20.00, 21.00, 22.00, 23.00, 24.00, 25.00, 26.00, 27.00, 28.00,
			   29.00, 30.00, 32.00, 34.00, 36.00};
//38.00, 40.00, 45.00, 50.00, 55.00, 60.00, 70.00, 80.00, 100.00};
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
  //double tot_mass = sum(pmf.mass, pmf.n);
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
  meshgraph_t *graph = tetramesh_meshgraph(pmf.tessellation->tetramesh);
  //printf("%f ms\n", get_time_ms() - t);

  //printf("Calling tetramesh_graph()...");
  t = get_time_ms();
  tetramesh_graph(pmf.tessellation->tetramesh);
  //printf("%f ms\n", get_time_ms() - t);

  //printf("Calling tetramesh_save_PLY_colors()...\n");

  //tetramesh_save_PLY_colors(pmf.tessellation->tetramesh, graph, "mesh.ply", colors);
  tetramesh_save_PLY(pmf.tessellation->tetramesh, graph, "mesh.ply");

  // print out the points
  //printf("pmf.points = [ ");
  //for (i = 0; i < pmf.n; i++)
  //  printf("%f %f %f %f ; ", pmf.points[i][0], pmf.points[i][1], pmf.points[i][2], pmf.points[i][3]);
  //printf("];\n");
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

  meshgraph_t *graph = tetramesh_meshgraph(pmf.tessellation->tetramesh);

  tetramesh_save_PLY_colors(pmf.tessellation->tetramesh, graph, "mres.ply", colors);
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


void test_bingham_mixture_sample(int argc, char *argv[])
{
  if (argc < 3) {
    printf("usage: %s <bmx_file> <n>\n", argv[0]);
    return;
  }

  int i, k, d = 4;
  bingham_mix_t *bmx = load_bmx(argv[1], &k);
  int n = atoi(argv[2]);

  double **X = new_matrix2(n, d);
  bingham_mixture_sample(X, &bmx[0], n);

  bingham_mix_t BM;
  bingham_cluster(&BM, X, n, d);

  for (i = 0; i < BM.n; i++) {
    print_bingham(&BM.B[i]);
    printf("---------------------------\n");
  }
}


void test_bingham_sample(int argc, char *argv[])
{
  if (argc < 5) {
    printf("usage: %s <z1> <z2> <z3> <num_samples>\n", argv[0]);
    exit(1);
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);
  int nsamples = atoi(argv[4]);

  double Z[3] = {z1, z2, z3};
  double V[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

  bingham_t B;
  bingham_new(&B, 4, Vp, Z);
  print_bingham(&B);

  printf("---------------------------\n");

  bingham_stats(&B);

  printf("Original scatter matrix:\n");
  int i, j, d=4;
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++)
      printf("%.4f, ", B.stats->scatter[i][j]);
    printf("\n");
  }

  double t0 = get_time_ms();
  double **X = new_matrix2(nsamples, 4);
  bingham_sample(X, &B, nsamples);
  printf("Sampled %d points in %.0f ms\n", nsamples, get_time_ms() - t0);

  bingham_fit(&B, X, nsamples, 4);

  print_bingham(&B);

  int n = nsamples;
  double **Xt = new_matrix2(d, n);
  transpose(Xt, X, n, d);
  double **S = new_matrix2(d, d);
  matrix_mult(S, Xt, X, d, n, d);
  mult(S[0], S[0], 1/(double)n, d*d);

  printf("Sample scatter matrix:\n");
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++)
      printf("%.4f, ", S[i][j]);
    printf("\n");
  }

}


void test_bingham_sample_pmf(int argc, char *argv[])
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
  bingham_sample_pmf(X, &pmf, nsamples);
  printf("Sampled %d points in %.0f ms\n", nsamples, get_time_ms() - t0);

  bingham_fit(&B, X, nsamples, 4);

  print_bingham(&B);
}


void test_bingham_compose(int argc, char *argv[])
{
  int i, j, d = 4;

  if (argc < 8) {
    printf("usage: %s <z11> <z12> <z13> <z21> <z22> <z23> <num_samples>\n", argv[0]);
    exit(1);
  }

  double z11 = atof(argv[1]);
  double z12 = atof(argv[2]);
  double z13 = atof(argv[3]);
  double z21 = atof(argv[4]);
  double z22 = atof(argv[5]);
  double z23 = atof(argv[6]);
  int nsamples = atoi(argv[7]);

  bingham_t B1;
  double Z1[3] = {z11, z12, z13};
  double V1[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp1[3] = {&V1[0][0], &V1[1][0], &V1[2][0]};
  bingham_new(&B1, 4, Vp1, Z1);
  bingham_stats(&B1);
  print_bingham(&B1);

  printf("S1:\n");
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++)
      printf("%.4f, ", B1.stats->scatter[i][j]);
    printf("\n");
  }

  printf("---------------------------\n");

  bingham_t B2;
  double Z2[3] = {z21, z22, z23};
  double V2[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};  //{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp2[3] = {&V2[0][0], &V2[1][0], &V2[2][0]};
  bingham_new(&B2, 4, Vp2, Z2);
  bingham_stats(&B2);
  print_bingham(&B2);

  printf("S2:\n");
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++)
      printf("%.4f, ", B2.stats->scatter[i][j]);
    printf("\n");
  }

  printf("---------------------------\n");

  // compose with method of moments
  //double **S_mom = new_matrix2(4,4);
  bingham_t B_mom;
  double t0 = get_time_ms();
  for (i = 0; i < nsamples; i++)
    bingham_compose(&B_mom, &B1, &B2);
  printf("Composed %d Bingham pairs with Method-of-Moments in %.0f ms\n", nsamples, get_time_ms() - t0);

  // compose with sampling
  t0 = get_time_ms();
  double **X1 = new_matrix2(nsamples, 4);
  bingham_sample(X1, &B1, nsamples);
  double **X2 = new_matrix2(nsamples, 4);
  bingham_sample(X2, &B2, nsamples);

  double **Y = new_matrix2(nsamples, 4);
  for (i = 0; i < nsamples; i++)
    quaternion_mult(Y[i], X1[i], X2[i]);

  int n = nsamples;
  double **Yt = new_matrix2(d, n);
  transpose(Yt, Y, n, d);
  double **S_sam = new_matrix2(d, d);
  matrix_mult(S_sam, Yt, Y, d, n, d);
  mult(S_sam[0], S_sam[0], 1/(double)n, d*d);
  printf("Composed 1 Bingham pair with %d samples in %.0f ms\n", nsamples, get_time_ms() - t0);

  printf("---------------------------\n");


  printf("Method-of-Moments scatter matrix:\n");
  bingham_stats(&B_mom);
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++)
      printf("%.4f, ", B_mom.stats->scatter[i][j]);
    printf("\n");
  }
  printf("---------------------------\n");
  printf("Sample scatter matrix:\n");
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++)
      printf("%.4f, ", S_sam[i][j]);
    printf("\n");
  }

  printf("\n---------------------------\n\n");

  t0 = get_time_ms();
  for (i = 0; i < nsamples; i++)
    bingham_compose_true_pdf(X1[i], &B1, &B2);
  printf("Computed %d true composition pdf's in %.0f ms\n", nsamples, get_time_ms() - t0);

  // compute sample mean error
  double tot_err = 0;
  for (i = 0; i < nsamples; i++) {
    double f_true = bingham_compose_true_pdf(X1[i], &B1, &B2);
    double f_approx = bingham_pdf(X1[i], &B_mom);
    //printf("f_true = %f, f_approx = %f\n", f_true, f_approx);
    tot_err += 2*fabs((f_true - f_approx) / (f_true + f_approx));
  }
  printf("mean sample err = %.2f%%\n", 100*tot_err/nsamples);

  // compute KL divergence
  printf("KL divergence = %f\n", bingham_compose_error(&B1, &B2));
}


void test_bingham_compose_multi(int argc, char *argv[])
{
  int a, b, d = 4;
  bingham_t B1, B2;
  double V1[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp1[3] = {&V1[0][0], &V1[1][0], &V1[2][0]};
  double V2[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double *Vp2[3] = {&V2[0][0], &V2[1][0], &V2[2][0]};
  double z_range[10] = {-100, -81, -64, -49, -36, -25, -16, -9, -4, -1};

  /*------------ isotropic-isotropic ------------//
  printf("ISO_ISO = [ ...\n");
  for (a = 0; a < 10; a++) {
    fprintf(stderr, "\n");
    for (b = 0; b < 10; b++) {
      fprintf(stderr, ".");
      double z1 = z_range[a];
      double z2 = z_range[b];
      double Z1[3] = {z1, z1, z1};
      double Z2[3] = {z2, z2, z2};
      bingham_new(&B1, d, Vp1, Z1);
      bingham_new(&B2, d, Vp2, Z2);
      
      printf("%f, %f, %f; ...\n", z1, z2, bingham_compose_error(&B1, &B2));
    }
  }
  printf("];\n\n\n");
  */

  //------------ anisotropic-anisotropic ------------//
  printf("ANI_ANI = [ ...\n");
  for (a = 0; a < 10; a++) {
    fprintf(stderr, "\n");
    for (b = a; b < 10; b++) {
      fprintf(stderr, ".");
      double z1 = z_range[a];
      double z2 = z_range[9-b];
      double Z1[3] = {-100, -100, z2};
      double Z2[3] = {-100, z1, z2};
      bingham_new(&B1, d, Vp1, Z1);
      bingham_new(&B2, d, Vp2, Z2);
      
      printf("%f, %f, %f; ...\n", z1, z2, bingham_compose_error(&B1, &B2));
    }
  }
  printf("];\n\n\n");  
}


void test_bingham_mult(int argc, char *argv[])
{
  if (argc < 7) {
    printf("usage: %s <z11> <z12> <z13> <z21> <z22> <z23> \n", argv[0]);
    exit(1);
  }

  double z11 = atof(argv[1]);
  double z12 = atof(argv[2]);
  double z13 = atof(argv[3]);
  double z21 = atof(argv[4]);
  double z22 = atof(argv[5]);
  double z23 = atof(argv[6]);
 
  double Z1[3] = {z11, z12, z13};
  double V1[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp1[3] = {&V1[0][0], &V1[1][0], &V1[2][0]};
  bingham_t B1;
  B1.d = 4;
  B1.Z = Z1;
  B1.V = Vp1;
  //bingham_new(&B1, 4, Vp1, Z1);

  double Z2[3] = {z21, z22, z23};
  double V2[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  //double V2[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp2[3] = {&V2[0][0], &V2[1][0], &V2[2][0]};
  bingham_t B2;
  B2.d = 4;
  B2.Z = Z2;
  B2.V = Vp2;
  //bingham_new(&B2, 4, Vp2, Z2);

  double Z[3];
  double V[3][4];
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};
  bingham_t B;
  B.Z = Z;
  B.V = Vp;
  //bingham_new(&B, 4, Vp, Z);

  double t0 = get_time_ms();
  int i, n=10000;
  for (i = 0; i < n; i++)
    bingham_mult(&B, &B1, &B2);
  double t1 = get_time_ms();
  printf("Performed %d same Bingham multiplications in %.0f ms\n", n, t1-t0);

  printf("B.F = %f\n", B.F);
  printf("B.Z = [%f %f %f]\n", B.Z[0], B.Z[1], B.Z[2]);
  printf("B.V[0] = [%f %f %f %f]\n", B.V[0][0], B.V[0][1], B.V[0][2], B.V[0][3]);
  printf("B.V[1] = [%f %f %f %f]\n", B.V[1][0], B.V[1][1], B.V[1][2], B.V[1][3]);
  printf("B.V[2] = [%f %f %f %f]\n", B.V[2][0], B.V[2][1], B.V[2][2], B.V[2][3]);
}


void test_bingham_mixture_mult(int argc, char *argv[])
{
  bingham_t B0[2];
  double w0[2] = {.8, .2};
  double Z00[3] = {-100, -1, -1};
  double V00[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp00[3] = {&V00[0][0], &V00[1][0], &V00[2][0]};
  B0[0].d = 4;
  B0[0].Z = Z00;
  B0[0].V = Vp00;
  double Z01[3] = {-100, -1, -1};
  double V01[3][4] = {{0,0,0,1}, {0,1,0,0}, {0,0,1,0}};
  double *Vp01[3] = {&V01[0][0], &V01[1][0], &V01[2][0]};
  B0[1].d = 4;
  B0[1].Z = Z01;
  B0[1].V = Vp01;

  bingham_t B1[2];
  double w1[2] = {.5, .5};
  double Z10[3] = {-100, -1, -1};
  double V10[3][4] = {{0,1,0,0}, {1,0,0,0}, {0,0,1,0}};
  double *Vp10[3] = {&V10[0][0], &V10[1][0], &V10[2][0]};
  B1[0].d = 4;
  B1[0].Z = Z10;
  B1[0].V = Vp10;
  double Z11[3] = {-1, -1, -1};
  double V11[3][4] = {{0,1,0,0}, {0,0,0,1}, {0,0,1,0}};
  double *Vp11[3] = {&V11[0][0], &V11[1][0], &V11[2][0]};
  B1[1].d = 4;
  B1[1].Z = Z11;
  B1[1].V = Vp11;

  bingham_mix_t BM0, BM1, BM;
  BM0.n = 2;
  BM0.w = w0;
  BM0.B = B0;
  BM1.n = 2;
  BM1.w = w1;
  BM1.B = B1;

  double t0 = get_time_ms();
  int i, n=10000;
  for (i = 0; i < n; i++)
    bingham_mixture_mult(&BM, &BM0, &BM1);
  double t1 = get_time_ms();
  printf("Performed %d same Bingham mixture multiplications in %.0f ms\n", n, t1-t0);

  printf("BM.n = %d\n", BM.n);
  printf("BM.w = [ ");
  for (i = 0; i < BM.n; i++)
    printf("%f ", BM.w[i]);
  printf("]\n");
  for (i = 0; i < BM.n; i++) {
    bingham_t *B = &BM.B[i];
    printf("B[%d]->F = %f\n", i, B->F);
    printf("B[%d]->Z = [%f %f %f]\n", i, B->Z[0], B->Z[1], B->Z[2]);
    printf("B[%d]->V[0] = [%f %f %f %f]\n", i, B->V[0][0], B->V[0][1], B->V[0][2], B->V[0][3]);
    printf("B[%d]->V[1] = [%f %f %f %f]\n", i, B->V[1][0], B->V[1][1], B->V[1][2], B->V[1][3]);
    printf("B[%d]->V[2] = [%f %f %f %f]\n", i, B->V[2][0], B->V[2][1], B->V[2][2], B->V[2][3]);
  }

  printf("\n\n");
}


void test_bingham_mixture_thresh_peaks(int argc, char *argv[])
{
  bingham_t B0[2];
  double w0[2] = {.8, .2};
  double Z00[3] = {-100, -1, -1};
  double V00[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp00[3] = {&V00[0][0], &V00[1][0], &V00[2][0]};
  B0[0].d = 4;
  B0[0].Z = Z00;
  B0[0].V = Vp00;
  double Z01[3] = {-100, -1, -1};
  double V01[3][4] = {{0,0,0,1}, {0,1,0,0}, {0,0,1,0}};
  double *Vp01[3] = {&V01[0][0], &V01[1][0], &V01[2][0]};
  B0[1].d = 4;
  B0[1].Z = Z01;
  B0[1].V = Vp01;

  bingham_t B1[2];
  double w1[2] = {.5, .5};
  double Z10[3] = {-100, -1, -1};
  double V10[3][4] = {{0,1,0,0}, {1,0,0,0}, {0,0,1,0}};
  double *Vp10[3] = {&V10[0][0], &V10[1][0], &V10[2][0]};
  B1[0].d = 4;
  B1[0].Z = Z10;
  B1[0].V = Vp10;
  double Z11[3] = {-1, -1, -1};
  double V11[3][4] = {{0,1,0,0}, {0,0,0,1}, {0,0,1,0}};
  double *Vp11[3] = {&V11[0][0], &V11[1][0], &V11[2][0]};
  B1[1].d = 4;
  B1[1].Z = Z11;
  B1[1].V = Vp11;

  bingham_mix_t BM0, BM1, BM;
  BM0.n = 2;
  BM0.w = w0;
  BM0.B = B0;
  BM1.n = 2;
  BM1.w = w1;
  BM1.B = B1;

  bingham_mixture_mult(&BM, &BM0, &BM1);

  int i;
  double peak, max_peak = 0;
  for (i = 0; i < BM.n; i++) {
    peak = BM.w[i] / BM.B[i].F;
    if (peak > max_peak)
      max_peak = peak;
  }

  printf("max_peak = %f\n", max_peak);

  double t0 = get_time_ms();
  int n=1000000;
  for (i = 0; i < n; i++)
    bingham_mixture_thresh_peaks(&BM, max_peak/10.0);
  double t1 = get_time_ms();
  printf("Performed %d same Bingham mixture thresh peaks in %.0f ms\n", n, t1-t0);

  printf("BM.n = %d\n", BM.n);
  printf("BM.w = [ ");
  for (i = 0; i < BM.n; i++)
    printf("%f ", BM.w[i]);
  printf("]\n");
  for (i = 0; i < BM.n; i++) {
    bingham_t *B = &BM.B[i];
    printf("B[%d]->F = %f\n", i, B->F);
    printf("B[%d]->Z = [%f %f %f]\n", i, B->Z[0], B->Z[1], B->Z[2]);
    printf("B[%d]->V[0] = [%f %f %f %f]\n", i, B->V[0][0], B->V[0][1], B->V[0][2], B->V[0][3]);
    printf("B[%d]->V[1] = [%f %f %f %f]\n", i, B->V[1][0], B->V[1][1], B->V[1][2], B->V[1][3]);
    printf("B[%d]->V[2] = [%f %f %f %f]\n", i, B->V[2][0], B->V[2][1], B->V[2][2], B->V[2][3]);
  }
}


void test_bingham_F_lookup_3d(int argc, char *argv[])
{
  if (argc < 4) {
    printf("usage: %s <z1> <z2> <z3>\n", argv[0]);
    exit(1);
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);

  double Z[3] = {z1, z2, z3};

  double t0 = get_time_ms();
  int i, n=1000000;
  for (i = 0; i < n; i++)
    bingham_F_lookup_3d(Z);
  double t1 = get_time_ms();
  printf("Performed %d same F-lookups in %.0f ms\n", n, t1-t0);

  t0 = get_time_ms();
  int j, k, m=200;
  n=0;
  for (i = 1; i <= m; i++) {
    for (j = 1; j <= i; j++) {
      for (k = 1; k <= j; k++) {
	Z[0] = -i;
	Z[1] = -j;
	Z[2] = -k;
	bingham_F_lookup_3d(Z);
	n++;
      }
    }
  }
  t1 = get_time_ms();
  printf("Performed %d unique F-lookups in %.0f ms\n", n, t1-t0);



  //double F_interp = bingham_F_lookup_3d(Z);
  //double F_series = bingham_F_3d(z1, z2, z3);
  //double error = F_interp - F_series;

  //printf("\nF_interp = %f, F_series = %f, error = %f\n\n", F_interp, F_series, error);
}


void test_bingham_sample_ridge(int argc, char *argv[])
{
  if (argc < 6) {
    printf("usage: %s <z1> <z2> <z3> <num_samples> <pthresh>\n", argv[0]);
    exit(1);
  }

  double z1 = atof(argv[1]);
  double z2 = atof(argv[2]);
  double z3 = atof(argv[3]);
  int nsamples = atoi(argv[4]);
  double pthresh = atof(argv[5]);

  double Z[3] = {z1, z2, z3};
  double V[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};

  bingham_t B;
  bingham_new(&B, 4, Vp, Z);
  //printf("B.F = %f\n", B.F);

  double t0 = get_time_ms();
  double **X = new_matrix2(nsamples, 4);
  bingham_sample_ridge(X, &B, nsamples, pthresh);
  printf("Sampled %d points in %.0f ms\n", nsamples, get_time_ms() - t0);

  //printf("X = [ ...\n");
  //int i;
  //for (i = 0; i < nsamples; i++) {
  //  printf("%f, %f, %f, %f ; ...\n", X[i][0], X[i][1], X[i][2], X[i][3]);
  //}
  //printf("];\n\n");
}


void test_bingham_stats(int argc, char *argv[])
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

  print_bingham(&B);

  int i, j, n = 1; //1000000;
  double t0 = get_time_ms();
  for (i = 0; i < n; i++)
    bingham_stats(&B);
  printf("Computed stats %d times in %.0f ms\n", n, get_time_ms() - t0);

  printf("B.stats->mode = [ %f %f %f %f ]\n", B.stats->mode[0], B.stats->mode[1], B.stats->mode[2], B.stats->mode[3]);
  printf("B.stats->dF = [ %f %f %f ]\n", B.stats->dF[0], B.stats->dF[1], B.stats->dF[2]);
  printf("B.stats->entropy = %f\n", B.stats->entropy);
  for (i = 0; i < B.d; i++) {
    printf("B.stats->scatter[%d] = [ ", i);
    for (j = 0; j < B.d; j++)
      printf("%f ", B.stats->scatter[i][j]);
    printf("]\n");
  }

  int yi = 9;
  printf("bingham_F_table[%d][%d][%d] = %f\n", yi, yi, yi, bingham_F_table_get(yi,yi,yi));
  printf("bingham_dF1_table[%d][%d][%d] = %f\n", yi, yi, yi, bingham_dF1_table_get(yi,yi,yi));
  printf("bingham_dF2_table[%d][%d][%d] = %f\n", yi, yi, yi, bingham_dF2_table_get(yi,yi,yi));
  printf("bingham_dF3_table[%d][%d][%d] = %f\n", yi, yi, yi, bingham_dF3_table_get(yi,yi,yi));

  bingham_fit_scatter(&B, B.stats->scatter, B.d);

  print_bingham(&B);
}


void test_bingham_KL_divergence(int argc, char *argv[])
{
  if (argc < 8) {
    printf("usage: %s <z11> <z12> <z13> <z21> <z22> <z23> <theta>\n", argv[0]);
    exit(1);
  }

  double z11 = atof(argv[1]);
  double z12 = atof(argv[2]);
  double z13 = atof(argv[3]);
  double z21 = atof(argv[4]);
  double z22 = atof(argv[5]);
  double z23 = atof(argv[6]);
  double theta = atof(argv[7]);
 
  double Z1[3] = {z11, z12, z13};
  double V1[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp1[3] = {&V1[0][0], &V1[1][0], &V1[2][0]};
  bingham_t B1;
  B1.d = 4;
  B1.Z = Z1;
  B1.V = Vp1;
  B1.F = bingham_F_lookup_3d(Z1);
  //bingham_new(&B1, 4, Vp1, Z1);

  double Z2[3] = {z21, z22, z23};
  //double V2[3][4] = {{0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  double V2[3][4] = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}};
  double *Vp2[3] = {&V2[0][0], &V2[1][0], &V2[2][0]};
  bingham_t B2;
  B2.d = 4;
  B2.Z = Z2;
  B2.V = Vp2;
  B2.F = bingham_F_lookup_3d(Z2);
  //bingham_new(&B2, 4, Vp2, Z2);

  // rotate B2 by theta about (1,0,0,0)
  double q[4] = {cos(theta/2.0), sin(theta/2.0), 0, 0};
  quaternion_mult(B2.V[0], q, B2.V[0]);
  quaternion_mult(B2.V[1], q, B2.V[1]);
  quaternion_mult(B2.V[2], q, B2.V[2]);

  printf("B2:\n");
  print_bingham(&B2);
  printf("\n");

  bingham_stats(&B1);
  bingham_stats(&B2);

  int i, j;
  printf("\n");
  printf("B1.stats->mode = [ %f %f %f %f ]\n", B1.stats->mode[0], B1.stats->mode[1], B1.stats->mode[2], B1.stats->mode[3]);
  printf("B1.stats->dF = [ %f %f %f ]\n", B1.stats->dF[0], B1.stats->dF[1], B1.stats->dF[2]);
  printf("B1.stats->entropy = %f\n", B1.stats->entropy);
  for (i = 0; i < B1.d; i++) {
    printf("B1.stats->scatter[%d] = [ ", i);
    for (j = 0; j < B1.d; j++)
      printf("%f ", B1.stats->scatter[i][j]);
    printf("]\n");
  }
  printf("\n");
  printf("B2.stats->mode = [ %f %f %f %f ]\n", B2.stats->mode[0], B2.stats->mode[1], B2.stats->mode[2], B2.stats->mode[3]);
  printf("B2.stats->dF = [ %f %f %f ]\n", B2.stats->dF[0], B2.stats->dF[1], B2.stats->dF[2]);
  printf("B2.stats->entropy = %f\n", B2.stats->entropy);
  for (i = 0; i < B2.d; i++) {
    printf("B2.stats->scatter[%d] = [ ", i);
    for (j = 0; j < B2.d; j++)
      printf("%f ", B2.stats->scatter[i][j]);
    printf("]\n");
  }


  int n = 1000;
  double t0 = get_time_ms();
  double d_KL;
  for (i = 0; i < n; i++)
    d_KL = bingham_KL_divergence(&B1, &B2);
  printf("Computed KL divergence %d times in %.0f ms\n", n, get_time_ms() - t0);
  printf("d_KL = %f\n", d_KL);

  bingham_t B;
  bingham_merge(&B, &B1, &B2, .5);
  bingham_stats(&B);

  printf("\nMerged binghams:\n");
  print_bingham(&B);

  printf("\n");
  printf("B.stats->mode = [ %f %f %f %f ]\n", B.stats->mode[0], B.stats->mode[1], B.stats->mode[2], B.stats->mode[3]);
  printf("B.stats->dF = [ %f %f %f ]\n", B.stats->dF[0], B.stats->dF[1], B.stats->dF[2]);
  printf("B.stats->entropy = %f\n", B.stats->entropy);
  for (i = 0; i < B.d; i++) {
    printf("B.stats->scatter[%d] = [ ", i);
    for (j = 0; j < B.d; j++)
      printf("%f ", B.stats->scatter[i][j]);
    printf("]\n");
  }
}


void test_bingham_init()
{
  double t0 = get_time_ms();

  bingham_init();

  double t1 = get_time_ms();

  fprintf(stderr, "Initialized bingham library in %.0f ms\n", t1-t0);
}


int main(int argc, char *argv[])
{
  test_bingham_init();

  //test_bingham_compose_multi(argc, argv);
  //test_bingham_compose(argc, argv);
  //test_bingham_stats(argc, argv);
  //test_bingham_KL_divergence(argc, argv);

  //test_bingham_mixture_mult(argc, argv);
  //test_bingham_mixture_thresh_peaks(argc, argv);
  test_bingham_mult(argc, argv);
  //test_bingham_F_lookup_3d(argc, argv);

  //test_bingham_mixture_sample(argc, argv);
  //test_bingham_sample(argc, argv);
  //test_bingham_sample_pmf(argc, argv);
  //test_bingham_sample_ridge(argc, argv);

  //test_fit_quaternions(argc, argv);
  //test_bingham_discretize(argc, argv);
  //test_bingham(argc, argv);
  //compute_bingham_constants(argc, argv);
  //test_bingham_pdf(argc, argv);
  //test_fit(argc, argv);


  return 0;
}
