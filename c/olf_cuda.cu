#include "cuda.h"
#include "include/bingham/cuda_wrapper.h"
//#include "cuda_profiler_api.h"
#include "bingham/olf.h"
#include "curand.h"

#include <math.h>

//#define CUDA_LAUNCH_BLOCKING 1

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

#define cu_free(x, msg) do{ if (cudaFree(x) != cudaSuccess) printf(msg); } while (0)

__device__ int first = 1;
__device__ __constant__ int big_primes[100] = {996311, 163573, 481123, 187219, 963323, 103769, 786979, 826363, 874891, 168991, 442501, 318679, 810377, 471073, 914519, 251059, 321983, 220009, 211877, 875339, 605603, 578483, 219619, 860089, 644911, 398819, 544927, 444043, 161717, 301447, 201329, 252731, 301463, 458207, 140053, 906713, 946487, 524389, 522857, 387151, 904283, 415213, 191047, 791543, 433337, 302989, 445853, 178859, 208499, 943589, 957331, 601291, 148439, 296801, 400657, 829637, 112337, 134707, 240047, 669667, 746287, 668243, 488329, 575611, 350219, 758449, 257053, 704287, 252283, 414539, 647771, 791201, 166031, 931313, 787021, 520529, 474667, 484361, 358907, 540271, 542251, 825829, 804709, 664843, 423347, 820367, 562577, 398347, 940349, 880603, 578267, 644783, 611833, 273001, 354329, 506101, 292837, 851017, 262103, 288989};

__device__ __constant__ double b_SR[3] = {0.2878,    -5.6214,      7.7247};
__device__ __constant__ double b_SN[3] = {0.1521,    -7.1290,     10.7090};
__device__ __constant__ double b_SL[3] = {0.2238,    -5.1827,      6.8242};
__device__ __constant__ double b_SA[3] = {0.1618,    -6.3992,      8.0207};
__device__ __constant__ double b_SB[3] = {0.2313,    -6.3463,      8.0651};

__device__ __constant__ double b_ER[3] = {0.3036,     0.2607,   -125.8843};
__device__ __constant__ double b_EN[3] = {0.1246,     1.4406,   -185.8350};
__device__ __constant__ double b_EL[3] = {0.2461,     0.2624,   -140.0192};
__device__ __constant__ double b_EA[3] = {0.1494,     0.2114,   -139.4324};
__device__ __constant__ double b_EB[3] = {0.2165,     0.2600,   -135.5203};


// util.h stuff **********************************
/*__device__ static void init_rand() {
  if (first) {
    first = 0;
    srand (time(NULL));
  }
  } */

curandGenerator_t gen;

__device__ void cu_randperm(int *x, int n, int d, uint r1, uint r2) {
  int i;
  if (d > n) {
    printf("Error: d > n in randperm()\n");
    return;
  }
  
  // sample a random starting point
  int i0 = r1 % n;

  // use a random prime step to cycle through x

  int step = big_primes[r2 % 100];

  int idx = i0;
  for (i = 0; i < d; i++) {
    x[i] = idx;
    idx = (idx + step) % n;
  }
}

// computes the max of x
int arr_max_i(int *x, int n)
{
  int i;

  int y = x[0];
  for (i = 1; i < n; i++)
    if (x[i] > y)
      y = x[i];

  return y;
}

// create a new n-by-m 2d matrix of doubles
__device__ double **cu_new_matrix2(int n, int m)
{
  if (n*m == 0) return NULL;
  int i;
  double *raw, **X;
  raw = (double *) malloc(n*m*sizeof(double));
  memset(raw, 0, n*m*sizeof(double));
  X = (double **) malloc(n * sizeof(double*));
  for (i = 0; i < n; i++)
    X[i] = raw + m*i;
  return X;
}

// free a 2d matrix of doubles
__device__ void cu_free_matrix2(double **X)
{
  if (X == NULL) return;
  free(X[0]);
  free(X);
}

// computes the dot product of z and y
__device__ double cu_dot(double x[], double y[], int n) {
  int i;
  double z = 0.0;
  for (i = 0; i < n; i++)
    z += x[i]*y[i];
  return z;
}

__device__ void cu_quaternion_to_rotation_matrix(double R[][3], double q[]) {
  double a = q[0];
  double b = q[1];
  double c = q[2];
  double d = q[3];

  R[0][0] = a*a + b*b - c*c - d*d;
  R[0][1] = 2*b*c - 2*a*d;
  R[0][2] = 2*b*d + 2*a*c;
  R[1][0] = 2*b*c + 2*a*d;
  R[1][1] = a*a - b*b + c*c - d*d;
  R[1][2] = 2*c*d - 2*a*b;
  R[2][0] = 2*b*d - 2*a*c;
  R[2][1] = 2*c*d + 2*a*b;
  R[2][2] = a*a - b*b - c*c + d*d;
}

__device__ void cu_matrix_vec_mult_3(double *y, double A[][3], double *x, int n, int m) {
  int i;
  /*if (y == x) {
    double *z = (double *) malloc(m * sizeof(double));
    memcpy(z, x, m*sizeof(double));
    for (i = 0; i < n; i++)
      y[i] = cu_dot(A[i], z, m);
    free(z);
  }
  else*/
  if (y == x) {
    printf("**************FIX CU_MATRIX_VEC_MULT CALL!\n");
  }
  for (i = 0; i < n; i++)
    y[i] = cu_dot(A[i], x, m);
}

__device__ void cu_matrix_vec_mult_flat(double *y, double *A, double *x, int n, int m) {
  int i;
  /*if (y == x) {
    double *z = (double *) malloc(m * sizeof(double));
    memcpy(z, x, m*sizeof(double));
    for (i = 0; i < n; i++)
      y[i] = cu_dot(A[i], z, m);
    free(z);
  }
  else*/
  if (y == x) {
    printf("**************FIX CU_MATRIX_VEC_MULT CALL!\n");
  }
  for (i = 0; i < n; i++)
    y[i] = cu_dot(&A[i * m], x, m);
}

// adds two vectors, z = x+y
__device__ void cu_add(double z[], double x[], double y[], int n) {
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i] + y[i];
}

// subtracts two vectors, z = x-y
__device__ void cu_sub(double z[], double x[], double y[], int n)
{
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i] - y[i];
}

// computes the sum of x's elements
__device__ double cu_sum(double x[], int n)
{
  int i;
  double y = 0;
  for (i = 0; i < n; i++)
    y += x[i];
  return y;
}

__device__ double cu_norm(double x[], int n) {
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += x[i]*x[i];

  return sqrt(d);
}

// compute the pdf of a normal random variable
__device__ double cu_normpdf(double x, double mu, double sigma)
{
  double dx = x - mu;

  return exp(-dx*dx / (2*sigma*sigma)) / (sqrt(2*M_PI) * sigma);
}

// sets y = x/sum(x)
__device__ void cu_normalize_pmf(double y[], double x[], int n)
{
  double d = cu_sum(x, n);
  int i;
  for (i = 0; i < n; i++)
    y[i] = x[i]/d;
}

// multiplies a vector by a scalar, y = c*x
__device__ void cu_mult(double y[], double x[], double c, int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] = c*x[i];
}

// sets y = x/norm(x)
__device__ void cu_normalize(double y[], double x[], int n)
{
  double d = cu_norm(x, n);
  int i;
  for (i = 0; i < n; i++)
    y[i] = x[i]/d;
}

// invert a quaternion
__device__ void cu_quaternion_inverse(double q_inv[4], double q[4])
{
  q_inv[0] = q[0];
  q_inv[1] = -q[1];
  q_inv[2] = -q[2];
  q_inv[3] = -q[3];
}

/*
 * get the plane equation coefficients (c[0]*x + c[1]*y + c[2]*z + c[3] = 0) from (point,normal)
 */
__device__ void cu_xyzn_to_plane(double *c, double *point, double *normal)
{
  c[0] = normal[0];
  c[1] = normal[1];
  c[2] = normal[2];
  c[3] = -cu_dot(point, normal, 3);
}

// reorder the rows of X, Y = X(idx,:)
__device__ void cu_reorder_rows(double *Y, double *X, int *idx, int n, int m)
{
  int i;
  if (X == Y) {
    printf("********** fix the call to cu_reorder_rows!\n");
  }
  for (i = 0; i < n; i++)
    memcpy(&Y[m*i], &X[m*idx[i]], m*sizeof(double));
}

__device__ double* get_row (cu_double_matrix_t *matrix, int i) {
  //return (double *)(((char *) matrix->ptr) + i * matrix->pitch);
  return &(matrix->ptr[i * matrix->m]);
}

__device__ double get_element(cu_double_matrix_t *matrix, int i, int j) {
  //double *row = (double *) (((char *) matrix->ptr) + i * matrix->pitch);
  //return *(row + j);
  return matrix->ptr[i * matrix->m + j];
}

/*
 * compute viewpoint (in model coordinates) for model placement (x,q) assuming observed viewpoint = (0,0,0)
 */
__device__ void cu_model_pose_to_viewpoint(double *vp, double x[], double q[])
{
  double q_inv[4];
  cu_quaternion_inverse(q_inv, q);
  double R_inv[3][3];
  cu_quaternion_to_rotation_matrix(R_inv,q_inv);
  cu_matrix_vec_mult_3(vp, R_inv, x, 3, 3);
  cu_mult(vp, vp, -1, 3);
}

__device__ void get_validation_points(int *idx, int num_points, int num_validation_points, uint *r)
{
  int i;
  if (num_validation_points == num_points)  // use all the points
    for (i = 0; i < num_validation_points; i++)
      idx[i] = i;
  else
    cu_randperm(idx, num_points, num_validation_points, r[0], r[1]);
}

__device__ void get_sub_cloud_at_pose(cu_double_matrix_t *points, double *cloud, int *idx, int n, double x[], double q[])
{
  double R[3][3];
  cu_quaternion_to_rotation_matrix(R,q);
  int i;
  double *row, *dest;
  for (i = 0; i < n; i++) {
    row = get_row(points, idx[i]);
    dest = &cloud[3*i];
    memcpy(dest, row, 3*sizeof(double));
    double tmp[3];
    cu_matrix_vec_mult_3(tmp, R, dest, 3, 3);
    cu_add(dest, tmp, x, 3);
  }
}

__device__ void get_sub_cloud_normals_rotated(cu_double_matrix_t *normals, double *cloud_normals, int *idx, int n, double q[])
{
  double R[3][3];
  cu_quaternion_to_rotation_matrix(R,q);
  int i;
  double *row, *dest;
  for (i = 0; i < n; i++) {
    row = get_row(normals, idx[i]);
    dest = &cloud_normals[3*i];
    double tmp[3];
    memcpy(tmp, row, 3*sizeof(double));
    cu_matrix_vec_mult_3(dest, R, tmp, 3, 3);
  }
}

__device__ void get_sub_cloud_lab(cu_double_matrix_t *lab, double *cloud_lab, int *idx, int n)
{
  cu_reorder_rows(cloud_lab, lab->ptr, idx, n, 3);
}

__device__ void range_image_xyz2sub(int *i, int *j, cu_range_image_data_t *range_image, double *xyz)
{
  //TODO: use range image viewpoint

  double d = cu_norm(xyz, 3);
  double x = atan2(xyz[0], xyz[2]);
  double y = acos(xyz[1] / d);

  int cx = (int)floor((x - range_image->min0) / range_image->res);
  int cy = (int)floor((y - range_image->min1) / range_image->res);

  *i = cx;
  *j = cy;

  if (!(cx>=0 && cy>=0 && (cx < range_image->w) && (cy < range_image->h))) {
    *i = -1;
    *j = -1;
  }
}

__device__ double compute_xyz_score(double *cloud, int *xi, int *yi, double *vis_pmf, scope_noise_model_t *noise_models, int num_validation_points, 
				    cu_double_matrix_t *range_image, cu_range_image_data_t *range_image_data, cu_int_matrix_t *range_image_cnt, scope_params_t *params, int score_round)
{
  double score = 0.0;
  //double range_sigma = params->range_sigma;
  //double dmax = 2*range_sigma;
  int i;
  for (i = 0; i < num_validation_points; i++) {
    if (vis_pmf[i] > .01/(double)num_validation_points) {
      double range_sigma = params->range_sigma * noise_models[i].range_sigma;
      double model_range = cu_norm(&cloud[3*i], 3);
      double dmax = 2*range_sigma;
      double dmin = dmax;
      int x, y;
      for (x = xi[i] - 1; x<=xi[i] + 1; ++x) {
	for (y = yi[i] - 1; y <= yi[i] + 1; ++y) {
	  if (x >= 0 && x < (range_image_data->w) && y>=0 && y<(range_image_data->h) && range_image_cnt->ptr[x * range_image_cnt->m + y] > 0) {
	    double obs_range = range_image->ptr[x * range_image->m + y];
	    double d = fabs(model_range - obs_range);
	    if (d < dmin) 
	      dmin = d;	    
	  }
	}
      }
      double d = dmin;
      score += vis_pmf[i] * log(cu_normpdf(d, 0, range_sigma));

    }
  }
  score -= log(cu_normpdf(0, 0, params->range_sigma));

  double w = 0;
  if (score_round == 2)
    w = params->score2_xyz_weight;
  else
    w = params->score3_xyz_weight;

  return w * score;
}

__device__ double compute_normal_score(double *cloud_normals, double *vis_pmf, scope_noise_model_t *noise_models, int num_validation_points, int *xi, int *yi,
				       cu_int_matrix_t *range_image_cnt, cu_double_matrix3d_t *range_image_normals, scope_params_t *params, int score_round)
{
  //TODO: make this a param
  double normalvar_thresh = .3;

  double score = 0.0;
  //double normal_sigma = params->normal_sigma;
  //double dmax = 2*normal_sigma;  // TODO: make this a param
  int i;
  double wtot = 0.0;
  for (i = 0; i < num_validation_points; i++) {
    if (vis_pmf[i] > .01/ (double) num_validation_points && noise_models[i].normal_sigma <= normalvar_thresh) {
      double normal_sigma = params->normal_sigma * noise_models[i].normal_sigma;
      double dmax = 2*normal_sigma;
      double d = dmax;
      if ((xi[i] != -1 && yi[i] != -1) && range_image_cnt->ptr[xi[i] * range_image_cnt->m + yi[i]] > 0) {
	// get distance from model normal to range image cell normal
	d = 1.0 - cu_dot(&cloud_normals[3*i], &(range_image_normals->ptr[xi[i] * range_image_normals->m * range_image_normals->p + yi[i] * range_image_normals->p]), 3);
	//d /= noise_models[i].normal_sigma;
	d = MIN(d, dmax);
      }
      score += vis_pmf[i] * log(cu_normpdf(d, 0, normal_sigma));
      wtot += vis_pmf[i];
    }
  }
  score /= wtot;
  score -= log(cu_normpdf(0, 0, params->normal_sigma));

  double w = 0;
  if (score_round == 2)
    w = params->score2_normal_weight;
  else
    w = params->score3_normal_weight;

  return w * score;
}

__device__ double compute_lab_score(int *xi, int *yi, double *lab, double *vis_pmf, scope_noise_model_t *noise_models, int n, cu_int_matrix_t *range_image_idx, cu_double_matrix_t *pcd_obs_bg_lab, 
				    scope_params_t *params, int score_round) 
{
  double scores[3] = {0, 0, 0};
  int i, j;
  //double L_weight = params->L_weight;
  //double lab_sigma = params->lab_sigma;
  //double dmax = 2*lab_sigma; // * sqrt(2.0 + L_weight*L_weight);  // TODO: make this a param
  for (i = 0; i < n; i++) {
    if (vis_pmf[i] > .01/(double)n) {
      //double d = dmax;
      double dlab[3], dmax[3], lab_sigma[3];
      for (j = 0; j < 3; j++) {
	lab_sigma[j] = params->lab_sigma * noise_models[i].lab_sigma[j];
	dmax[j] = 2*lab_sigma[j];
	dlab[j] = dmax[j];
      }
      int obs_idx = range_image_idx->ptr[xi[i] * range_image_idx->m + yi[i]];
      if (obs_idx >= 0) {
	double *obs_lab = &pcd_obs_bg_lab->ptr[obs_idx * pcd_obs_bg_lab->m];
	cu_sub(dlab, &lab[3*i], obs_lab, 3);
	//dlab[0] = L_weight * (lab[i][0] - obs_lab[0]) / noise_models[i].lab_sigma[0];
	//dlab[1] = (lab[i][1] - obs_lab[1]) / noise_models[i].lab_sigma[1];
	//dlab[2] = (lab[i][2] - obs_lab[2]) / noise_models[i].lab_sigma[2];
	//dlab[2] = 0;
	//d = norm(dlab, 3);
	//d = MIN(d, dmax);
	for (j = 0; j < 3; j++)	  
	  dlab[j] = MIN(dlab[j], dmax[j]);

	//dbug
	//if (params->verbose && (i%100==0)) {
	//  printf("model lab[%d] = [%.2f, %.2f, %.2f], obs_lab = [%.2f, %.2f, %.2f]\n", i, lab[i][0], lab[i][1], lab[i][2], obs_lab[0], obs_lab[1], obs_lab[2]);
	//}
      }
      //score += vis_pmf[i] * log(normpdf(d, 0, lab_sigma));
      for (j = 0; j < 3; j++)
	scores[j] += vis_pmf[i] * log(cu_normpdf(dlab[j], 0, lab_sigma[j]));

    }
  }
  //score -= log(normpdf(0, 0, params->lab_sigma));

  double lab_weights2[3] = {params->score2_L_weight, params->score2_A_weight, params->score2_B_weight};
  double lab_weights3[3] = {params->score3_L_weight, params->score3_A_weight, params->score3_B_weight};

  double *w = NULL;
  if (score_round == 2)
    w = lab_weights2;
  else
    w = lab_weights3;

  return cu_dot(scores, w, 3);
}

__device__ double compute_vis_score(double *vis_prob, int n, scope_params_t *params, int score_round)
{
  double score = log(cu_sum(vis_prob, n) / (double) n);

  double w = 0;
  if (score_round == 2)
    w = params->score2_vis_weight;
  else
    w = params->score3_vis_weight;

  return w * score;
}

/*void labdist_color_shift(double *shift, pcd_color_model_t *color_model, int *idx, int n, double **obs_lab, double *obs_weights, double pmin, scope_params_t *params)
{
  //TODO: make these params
  double lambda = 1.0;
  double shift_threshold = 0.1;

  double **C_inv = new_matrix2(3,3);
  double **B = new_matrix2(3,3);
  inv(B, color_model->avg_cov, 3);
  double **A = new_matrix2(3,3);
  double z[3];  // m-bar
  double w;

  memset(shift, 0, 3*sizeof(double));

  int i, j, iter, max_iter = 10;
  for (iter = 0; iter < max_iter; iter++) {

    // reset shift statistics
    memset(A[0], 0, 9*sizeof(double));
    memset(z, 0, 3*sizeof(double));
    w = 0;

    for (i = 0; i < n; i++) {

      if (obs_weights[i] == 0.0)
	continue;

      int cnt1 = color_model->cnts[0][idx[i]];
      int cnt2 = color_model->cnts[1][idx[i]];
      if (cnt1 < 4)
	cnt1 = 0;
      if (cnt2 < 4)
	cnt2 = 0;
      if (cnt1 == 0 && cnt2 == 0)
	continue;

      double *m1 = color_model->means[0][idx[i]];
      double *m2 = color_model->means[1][idx[i]];
      double **C1 = color_model->covs[0][idx[i]];
      double **C2 = color_model->covs[1][idx[i]];

      // assign observed color to a cluster
      double y[3];  // current obs_lab[i]
      add(y, obs_lab[i], shift, 3);
      double p1 = (cnt1 > 0 ? mvnpdf(y, m1, C1, 3) : 0);
      double p2 = (cnt2 > 0 ? mvnpdf(y, m2, C2, 3) : 0);
      
      // check if assigned cluster could be a specularity cluster (i.e., has higher L-value)
      if ((p1 > p2 && p2 > 0 && m1[0] > m2[0]) || (p2 > p1 && p1 > 0 && m2[0] > m1[0]))
	continue;

      double *m = (p1 > p2 ? m1 : m2);
      double **C = (p1 > p2 ? C1 : C2);

      double maxp = mvnpdf(m, m, C, 3);
      double p = mvnpdf(y, m, C, 3);

      // check if point is an outlier of the cluster
      if (p < pmin*maxp)
	continue;

      // add observed color and color model covariance matrix to the shift statistics
      for (j = 0; j < 3; j++)
	z[j] = z[j] + obs_weights[i]*(m[j] - obs_lab[i][j]);
      inv(C_inv, C, 3);
      for (j = 0; j < 9; j++)
	A[0][j] = A[0][j] + obs_weights[i]*C_inv[0][j];
      w += obs_weights[i];
    }

    mult(z, z, 1/w, 3);  // avg. z
    mult(A[0], A[0], lambda/w, 9);  // avg. A and multiply by lambda

    // solve for best shift = inv(lambda*A+B)*lambda*A*z
    double new_shift[3];
    matrix_vec_mult(z, A, z, 3, 3);
    add(A[0], A[0], B[0], 9);
    inv(C_inv, A, 3);
    matrix_vec_mult(new_shift, C_inv, z, 3, 3);
    double d2 = dist2(shift, new_shift, 3);
    memcpy(shift, new_shift, 3*sizeof(double));

    //printf("shift = [%f, %f, %f]\n", shift[0], shift[1], shift[2]);  //dbug

    if (d2 < shift_threshold*shift_threshold)
      break;
  }

  // apply shift to obs_lab
  for (i = 0; i < n; i++)
    if (obs_weights[i] > 0.0)
      add(obs_lab[i], obs_lab[i], shift, 3);

  free_matrix2(A);
  free_matrix2(B);
  free_matrix2(C_inv);
}

double compute_labdist_score(double **cloud, pcd_color_model_t *color_model, int *idx, double *vis_pmf, scope_noise_model_t *noise_models, int n,
			     range_image_t *obs_range_image, pcd_t *pcd_obs, scope_params_t *params, int score_round)
{
  //TODO: make this a param
  double pmin = .1;

  // get obs colors
  double **obs_lab = new_matrix2(n,3);
  double obs_weights[n];
  memset(obs_weights, 0, n*sizeof(double));
  int i;
  for (i = 0; i < n; i++) {
    if (vis_pmf[i] > .01/(double)n) {
      int xi,yi;
      range_image_xyz2sub(&xi, &yi, obs_range_image, cloud[i]);

      int obs_idx = obs_range_image->idx[xi][yi];
      if (obs_idx >= 0) {
	memcpy(obs_lab[i], pcd_obs->lab[obs_idx], 3*sizeof(double));
	obs_weights[i] = vis_pmf[i];
      }
    }
  }

  // get color shift (and apply it to obs_lab)
  double color_shift[3];
  labdist_color_shift(color_shift, color_model, idx, n, obs_lab, obs_weights, pmin, params);

  if (params->verbose) {
    memset(mps_labdist_p_ratios_, 0, n*sizeof(double));
  }

  double zero[3] = {0,0,0};
  double score = 0.0;
  for (i = 0; i < n; i++) {
    if (vis_pmf[i] > .01/(double)n) {
      double logp = labdist_likelihood(color_model, idx[i], (obs_weights[i] > 0 ? obs_lab[i] : zero), pmin, params);
      score += vis_pmf[i] * logp;
    }
  }

  double w = 0;
  if (score_round == 2)
    w = params->score2_labdist_weight;
  else
    w = params->score3_labdist_weight;

  free_matrix2(obs_lab);

  return w * score;
}*/

__device__ double compute_visibility_prob(double *point, double *normal, int xi, int yi, cu_range_image_data_t *ri_data, cu_double_matrix_t *range_image, double vis_thresh, int search_radius)
//double compute_visibility_prob(double *point, double *normal, range_image_t *obs_range_image, double vis_thresh, int search_radius)
{
  double V[3];
  cu_normalize(V, point, 3);

  if (normal != NULL && cu_dot(V, normal, 3) >= -.1)  // normals pointing away
    return 0.0;

  if (xi == -1 && yi == -1)
    return 0.0;

  double model_range = cu_norm(point, 3);
  double obs_range = range_image->ptr[xi * range_image->m + yi];

  if (search_radius > 0) {
    int x0 = MAX(xi - search_radius, 0);
    int x1 = MIN(xi + search_radius, ri_data->w - 1);
    int y0 = MAX(yi - search_radius, 0);
    int y1 = MIN(yi + search_radius, ri_data->h - 1);
    int x, y;
    for (x = x0; x <= x1; x++)
      for (y = y0; y <= y1; y++)
	obs_range = MAX(obs_range, range_image->ptr[x * range_image->m + y]);
  }

  double dR = model_range - obs_range;
  return (dR < 0 ? 1.0 : cu_normpdf(dR/vis_thresh, 0, 1) / .3989);  // .3989 = normpdf(0,0,1)
}

__device__ inline double cu_sigmoid(double x, const double *b)
{
  return b[0] + (1 - b[0]) / (1 + exp(-b[1]-b[2]*x));
}

__device__ void get_noise_models(scope_noise_model_t *noise_models, double *cloud, double *cloud_normals, double x[], double q[], int *idx, int n, 
				 cu_double_matrix_t *ved, cu_double_matrix_t *range_edges_model_views, cu_double_arr_t *normalvar)
{
  int i;

  // prep for lookup edge distances for closest model viewpoint
  double vp[3];

  cu_model_pose_to_viewpoint(vp, x, q);
  int vi;
  double vi_max = -(1<<29);
  // Did this without functions to avoid stuff like a[n] that's not supported in C++
  for (i = 0; i < range_edges_model_views->n; ++i) {
    double tmp = cu_dot(&range_edges_model_views->ptr[i * range_edges_model_views->m], vp, 3);
    if (tmp > vi_max) {
      vi = i;
      vi_max = tmp;
    }
  }

  double surface_angles, edge_dists;
  // compute sigmas
  for (i = 0; i < n; i++) {
    double normalized[3];
    cu_normalize(normalized, &cloud[3*i], 3);
    surface_angles = 1 + cu_dot(normalized, &cloud_normals[3*i], 3);
    edge_dists = ved->ptr[idx[i] * ved->m + vi];
    noise_models[i].range_sigma = .5*cu_sigmoid(surface_angles, b_SR) + .5*cu_sigmoid(edge_dists, b_ER);
    noise_models[i].normal_sigma = .5*cu_sigmoid(surface_angles, b_SN) + .5*cu_sigmoid(edge_dists, b_EN);
    noise_models[i].lab_sigma[0] = .5*cu_sigmoid(surface_angles, b_SL) + .5*cu_sigmoid(edge_dists, b_EL);
    noise_models[i].lab_sigma[1] = .5*cu_sigmoid(surface_angles, b_SA) + .5*cu_sigmoid(edge_dists, b_EA);
    noise_models[i].lab_sigma[2] = .5*cu_sigmoid(surface_angles, b_SB) + .5*cu_sigmoid(edge_dists, b_EB);

    noise_models[i].normal_sigma = MAX(noise_models[i].normal_sigma, normalvar->ptr[idx[i]]);
  }
  
}

__device__ void cu_transform_cloud(double *cloud2, double *cloud, int n, double x[], double q[])
{
  double R[3][3];
  cu_quaternion_to_rotation_matrix(R,q);
  int i;
  for (i = 0; i < n; i++) {
    double tmp[3];
    cu_matrix_vec_mult_3(tmp, R, &cloud[i*3], 3, 3);
    memcpy(&cloud2[3*i], tmp, 3*sizeof(double));
    if (x != NULL) {
      cu_add(&cloud2[i*3], &cloud2[i*3], x, 3);
    }
  }
}

__device__ void get_range_edge_points(double *P, int *idx, int *n_ptr, double x[], double q[], 
				      cu_double_matrix_t *range_edges_model_views, cu_int_arr_t *range_edges_view_cnt, cu_int_arr_t *range_edges_view_idx, cu_double_matrix_t *range_edges_points, uint *r)
{
  // compute viewpoint for model placement (x,q) assuming observed viewpoint = (0,0,0)
  double vp[3];
  cu_model_pose_to_viewpoint(vp, x, q);

  int i;
  double i_max = -(1<<29);
  // Did this without functions to avoid stuff like a[n] that's not supported in C++
  int ii;
  for (ii = 0; ii < range_edges_model_views->n; ++ii) {
    double tmp = cu_dot(&range_edges_model_views->ptr[ii * range_edges_model_views->m], vp, 3);
    if (tmp > i_max) {
      i = ii;
      i_max = tmp;
    }
  }
  
  int vp_idx = range_edges_view_idx->ptr[i];
  int num_edge_points = range_edges_view_cnt->ptr[i];

  //printf("vp = [%f, %f, %f], closest stored vp = [%f, %f, %f]\n", vp[0], vp[1], vp[2],
  //	 range_edges_model->views[i][0], range_edges_model->views[i][1], range_edges_model->views[i][2]);  //dbug

  // sample edge points to validate
  //int idx[num_edge_points];
  // TODO(sanja): This is a performance hit, figure out a way around it
  int n = *n_ptr;
  if (n >= num_edge_points || n == 0) {
    n = num_edge_points;
    for (i = 0; i < n; i++)
      idx[i] = i;
  }
  else
    cu_randperm(idx, num_edge_points, n, r[0], r[1]);

  // make idx be pcd point indices
  for (i = 0; i < n; i++)
    idx[i] += vp_idx;

  //printf("n = %d, idx[0] = %d, idx[n-1] = %d\n", n, idx[0], idx[n-1]); //dbug

  // get the actual points in the correct pose
  // TODO(sanja): performance hit...
  cu_reorder_rows(P, range_edges_points->ptr, idx, n, 3);

  *n_ptr = n;
}

__device__ void cu_get_sub_matrix(double *Y, double *X, int x0, int y0, int x1, int y1)
{
  int h = y1-y0+1;

  int x;
  for (x = x0; x <= x1; x++)
    memcpy(&Y[(x-x0) * h], &X[x * h + y0], h*sizeof(double));
}

__device__ void cu_dilate_matrix(double *Y, double *X, int n, int m, int n2, int m2)
{
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (X[i * m2 + j] > 0.0)
	Y[i * m + j] = X[i * m2 + j];
      else {
	int cnt = 0;
	double p = 0.0;
	if (i > 0 && X[(i-1) * m2 + j] > 0.0) {  p += X[(i-1)*m2 + j]; cnt++;  }
	if (i < n-1 && X[(i+1) * m2 + j] > 0.0) {  p += X[(i+1) * m2 + j]; cnt++;  }
	if (j > 0 && X[i*m2 + j-1] > 0.0) {  p += X[i*m2 + j-1]; cnt++;  }
	if (j < m-1 && X[i*m2 + j+1] > 0.0) {  p += X[i*m2 + j+1]; cnt++;  }
	if (cnt > 0)
	  Y[i * m + j] = p / (double)cnt;
      }
    }
  }
}

__device__ void compute_occ_edges(int *occ_edges, double *V, double *V2, int *num_occ_edges, int *xi, int *yi, double *vis_prob, int n, cu_range_image_data_t *ri_data, scope_params_t *params)
{
  // create vis_prob image, V
  int i;
  int w = ri_data->w;
  int h = ri_data->h;
  int x0 = w, y0 = h, x1 = 0, y1 = 0;  // bounding box for model points in vis_prob_image
  for (i = 0; i < n; i++) {
    if ((xi[i] != -1 && yi[i] != -1) && vis_prob[i] > V[xi[i] * h + yi[i]]) {
      V[xi[i] * h + yi[i]] = vis_prob[i];
      if (xi[i] < x0)
	x0 = xi[i];
      if (xi[i] > x1)
	x1 = xi[i];
      if (yi[i] < y0)
	y0 = yi[i];
      if (yi[i] > y1)
	y1 = yi[i];
    }
  }

  // downsample vis_prob sub matrix (loses a row or column if w2 or h2 is odd)
  cu_get_sub_matrix(V2, V, x0, y0, x1, y1);
  int w2 = (x1-x0+1)/2;
  int h2 = (y1-y0+1)/2;
  int w22 = x1-x0+1;
  int h22 = y1-y0+1;

  int x,y;
  for (x = 0; x < w2; x++) {
    for (y = 0; y < h2; y++) {
      double v2 = MAX(V2[2*x * h22 + 2*y], V2[(2*x+1) * h22 + 2*y]);
      v2 = MAX(v2, V2[2*x * h22 + (2*y+1)]);
      V2[x*w22 + y] = MAX(v2, V2[(2*x+1)*h22 + (2*y+1)]);
    }
  }

  // dilate vis_prob sub matrix
  cu_dilate_matrix(V, V2, w2, h2, w22, h22);
  //dilate_matrix(V2, V, w2, h2);
  //dilate_matrix(V, V2, w2, h2);
  //dilate_matrix(V2, V, w2, h2);
  //dilate_matrix(V, V2, w2, h2);

  // compute edges where vis_prob crosses .5 threshold
  int cnt=0;
  for (x = 0; x < w2-1; x++) {
    for (y = 0; y < h2-1; y++) {
      if (V[x*h + y] >= .5) {
	if ((x > 0 && V[(x-1) * h + y] > 0.0 && V[(x-1)*h + y] < .5) || (x < w2-1 && V[(x+1) * h + y] > 0.0 && V[(x+1) * h + y] < .5) ||
	    (y > 0 && V[x * h + y-1] > 0.0 && V[x * h + y-1] < .5) || (y < h2-1 && V[x * h + y+1] > 0.0 && V[x*h + y+1] < .5)) {
	  ++cnt;
	}
      }
    }
  }
  *num_occ_edges = cnt;

  if (cnt==0)
    return;

  cnt = 0;
  for (x = 0; x < w2-1; x++) {
    for (y = 0; y < h2-1; y++) {
      if (V[x*h + y] >= .5) {
	if ((x > 0 && V[(x-1) * h + y] > 0.0 && V[(x-1) *h + y] < .5) || (x < w2-1 && V[(x+1) * h + y] > 0.0 && V[(x+1) * h + y] < .5) ||
	    (y > 0 && V[x * h + y-1] > 0.0 && V[x * h + y-1] < .5) || (y < h2-1 && V[x * h + y+1] > 0.0 && V[x*h + y+1] < .5)) {
	  occ_edges[2*cnt] = x0 + 2*x;
	  occ_edges[2*cnt + 1] = y0 + 2*y;
	  ++cnt;
	}
      }
    }
  }
}

__device__ double compute_edge_score(double *P, double *vis_prob, double *vis_pmf, int n, int *occ_edges, int num_occ_edges, cu_range_image_data_t *range_image_data,
				     cu_double_matrix_t *range_image, cu_double_matrix_t *edge_image, scope_params_t *params, int score_round)
{
  if (n == 0)
    return 0.0;

  // compute visibility of sampled model edges
  int vis_pixel_radius = 2;
  int i;
  for (i = 0; i < n; i++) {
    int x, y;
    range_image_xyz2sub(&x, &y, range_image_data, &P[3*i]);
    if (x == -1 && y == -1) {
      vis_prob[i] = 0.0;
      continue;
    }
    vis_prob[i] = compute_visibility_prob(&P[3*i], NULL, x, y, range_image_data, range_image, params->vis_thresh, vis_pixel_radius);
  }
  cu_normalize_pmf(vis_pmf, vis_prob, n);

  // compute obs_edge_image score for sampled model edges
  double score = 0;
  int xi,yi;
  for (i = 0; i < n; i++) {
    range_image_xyz2sub(&xi, &yi, range_image_data, &P[3*i]);
    if (xi != -1 && yi != -1) {
      score += vis_pmf[i] * edge_image->ptr[xi*edge_image->m + yi];
    }
  }
  double vis_score = log(cu_sum(vis_prob, n) / (double) n);

  //printf("gpu %d %lf\n", n, vis_score);
    
  // add occlusion edges to score
  double occ_score = 0.0;
  if (num_occ_edges > 0) {
    for (i = 0; i < num_occ_edges; i++) {
      int x = occ_edges[2 * i];
      int y = occ_edges[2 * i + 1];
      occ_score += edge_image->ptr[x * edge_image->m + y];
    }
    occ_score /= (double) num_occ_edges;
    occ_score = num_occ_edges*occ_score / (double)(n + num_occ_edges);
    score = n*score / (double)(n + num_occ_edges);
  }

  double w1=0, w2=0, w3=0;
  if (score_round == 2) {
    w1 = params->score2_edge_weight;
    w2 = params->score2_edge_vis_weight;
    w3 = params->score2_edge_occ_weight;
  }
  else {
    w1 = params->score3_edge_weight;
    w2 = params->score3_edge_vis_weight;
    w3 = params->score3_edge_occ_weight;
  }

  return (w1 * score) + (w2 * vis_score) + (w3 * occ_score);
}

__device__ double cu_model_placement_score(double x[], double q[], cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, scope_params_t *cu_params, int score_round, 
					   int *xi, int *yi, int *idx, double *cloud, double *cloud_normals, double *cloud_lab, int num_validation_points, double *vis_prob, double *vis_pmf, uint *r,
					   scope_noise_model_t *noise_models,
					   int *idx_edge, double *P, double *V_edge, double *V2_edge, int *occ_edges, double *vis_prob_edge, double *vis_pmf_edge) {
  //int dbg_timed = 1;
  //double t0 = get_time_ms();  //dbug
  
  // get model validation points
  
  int i;
  get_validation_points(idx, cu_model->num_points, num_validation_points, r);
  
  /*if (dbg_timed) {
    printf("break 0, %.2f ms\n", get_time_ms() - t0);  //dbug
    t0 = get_time_ms();
    }*/
  
  /*if (dbg_timed) {
    printf("break 1, %.2f ms\n", get_time_ms() - t0);  //dbug
    t0 = get_time_ms();
    }*/
  
  // extract transformed model validation features
  get_sub_cloud_at_pose(&(cu_model->points), cloud, idx, num_validation_points, x, q);
  double *dest;
  for (i = 0; i < num_validation_points; ++i) {
    dest = &cloud[3*i];
    range_image_xyz2sub(&xi[i], &yi[i], &(cu_obs->range_image_data), dest);
  }
  
  if (score_round == 1) {  // after c=1, just use free space to score
    double dthresh = .05;  //TODO: make this a param
    double score = 0;

    for (i = 0; i < num_validation_points; i++) {
      dest = &cloud[3*i];
      if ((xi[i] != -1 && yi[i] != -1) && get_element(&(cu_obs->range_image), xi[i], yi[i]) > dthresh + cu_norm(dest, 3))
	score -= 1.0;
    }
    score /= (double)num_validation_points;
    return score;
  }
    
  get_sub_cloud_normals_rotated(&(cu_model->normals), cloud_normals, idx, num_validation_points, q);
  
  //double **cloud_sdw = get_sub_cloud_sdw(model_data->pcd_model, idx, num_validation_points, params);
  get_sub_cloud_lab(&(cu_model->lab), cloud_lab, idx, num_validation_points);
  //double **cloud_labdist = get_sub_cloud_labdist(model_data->pcd_model, idx, num_validation_points);
  //double **cloud_xyzn = get_xyzn_features(cloud, cloud_normals, num_validation_points, params);

  /*if (dbg_timed) {
    printf("break 2, %.2f ms\n", get_time_ms() - t0);  //dbug
    t0 = get_time_ms();
    }*/

  // compute p(visibile)
  for (i = 0; i < num_validation_points; i++) {
    vis_prob[i] = compute_visibility_prob(&cloud[3 * i], &cloud_normals[3 * i], xi[i], yi[i], &(cu_obs->range_image_data), &(cu_obs->range_image), cu_params->vis_thresh, 0);
  }
  cu_normalize_pmf(vis_pmf, vis_prob, num_validation_points);

  //if (params->verbose)
  //  memcpy(mps_vis_prob_, vis_prob, num_validation_points*sizeof(double));
  
  /*if (dbg_timed) {
    printf("break 3, %.2f ms\n", get_time_ms() - t0);  //dbug
    t0 = get_time_ms();
    }*/

  // compute noise models
  get_noise_models(noise_models, cloud, cloud_normals, x, q, idx, num_validation_points, &(cu_model->ved), &(cu_model->range_edges_model_views), &(cu_model->normalvar));

  /*
  if (dbg_timed) {
    printf("break 4, %.2f ms\n", get_time_ms() - t0);  //dbug
    t0 = get_time_ms();
    }*/
  
  // compute nearest neighbors
  //int nn_idx[num_validation_points];  memset(nn_idx, 0, num_validation_points*sizeof(int));
  //double nn_d2[num_validation_points];  memset(nn_d2, 0, num_validation_points*sizeof(double));
  //int search_radius = 0;  // pixels
  //for (i = 0; i < num_validation_points; i++)
  //if (vis_prob[i] > .01)
  //range_image_find_nn(&nn_idx[i], &nn_d2[i], &cloud[i], &cloud_xyzn[i], 1, 6, obs_xyzn, obs_range_image, search_radius);
  //range_image_find_nn(&nn_idx[i], &nn_d2[i], &cloud[i], &cloud[i], 1, 3, pcd_obs->points, obs_range_image, search_radius);
  
  double normal_score = compute_normal_score(cloud_normals, vis_pmf, noise_models, num_validation_points, xi, yi, &(cu_obs->range_image_cnt), &(cu_obs->range_image_normals), cu_params, score_round);
  double xyz_score = compute_xyz_score(cloud, xi, yi, vis_pmf, noise_models, num_validation_points, &(cu_obs->range_image), &(cu_obs->range_image_data), &(cu_obs->range_image_cnt), 
				       cu_params, score_round);
  double lab_score = compute_lab_score(xi, yi, cloud_lab, vis_pmf, noise_models, num_validation_points, &(cu_obs->range_image_idx), &(cu_obs->range_image_pcd_obs_bg_lab), cu_params, score_round);
  //double labdist_score = compute_labdist_score(cloud, cloud_labdist, vis_pmf, noise_models, num_validation_points, obs_data->obs_range_image, obs_data->pcd_obs_bg, params, score_round);
  double vis_score = compute_vis_score(vis_prob, num_validation_points, cu_params, score_round);

  /* ----- Sanja's comment --------
  double labdist_score = 0;
  if (round > 2)
    labdist_score = compute_labdist_score(cloud, model_data->color_model, idx, vis_pmf, noise_models, num_validation_points, obs_data->obs_range_image, obs_data->pcd_obs_bg, params, score_round);
  
  // get fpfh score (TODO: add fpfh features to occ_model)
  int fpfh_num_validation_points = (params->num_validation_points > 0 ? params->num_validation_points : model_data->fpfh_model->num_points);
  int fpfh_idx[fpfh_num_validation_points];
  get_validation_points(fpfh_idx, model_data->fpfh_model, fpfh_num_validation_points);
  double **fpfh_cloud = get_sub_cloud_at_pose(model_data->fpfh_model, fpfh_idx, fpfh_num_validation_points, x, q);
  double **fpfh_cloud_normals = get_sub_cloud_normals_rotated(model_data->fpfh_model, fpfh_idx, fpfh_num_validation_points, q);
  double **fpfh_cloud_f = get_sub_cloud_fpfh(model_data->fpfh_model, fpfh_idx, fpfh_num_validation_points);
  double fpfh_vis_prob[fpfh_num_validation_points];
  for (i = 0; i < fpfh_num_validation_points; i++)
    fpfh_vis_prob[i] = compute_visibility_prob(fpfh_cloud[i], fpfh_cloud_normals[i], obs_data->obs_range_image, params->vis_thresh, 0);
  double fpfh_vis_pmf[fpfh_num_validation_points];
  normalize_pmf(fpfh_vis_pmf, fpfh_vis_prob, fpfh_num_validation_points);
  double fpfh_score = compute_fpfh_score(fpfh_cloud, fpfh_cloud_f, fpfh_vis_pmf, fpfh_num_validation_points, obs_data->obs_fg_range_image, obs_data->pcd_obs, params, score_round);
  
  double fpfh_score = 0;
  //double xyzn_score = compute_xyzn_score(nn_d2, vis_pmf, num_validation_points, params);
  //double xyz_score = compute_xyz_score(cloud, nn_idx, vis_pmf, num_validation_points, pcd_obs, params);
  //double sdw_score = compute_sdw_score(cloud_sdw, nn_idx, vis_pmf, num_validation_points, pcd_obs, params);
  //double lab_score = compute_lab_score(cloud_lab, nn_idx, vis_pmf, num_validation_points, pcd_obs, params);

  if (dbg_timed) {
    printf("break 5, %.2f ms\n", get_time_ms() - t0);  //dbug
    t0 = get_time_ms();
  }
  */ // End Sanja's comment
  
  //TODO: move this to compute_edge_score()
  double edge_score = 0.0;
  if (cu_obs->edge_image.ptr) {
    int n = cu_params->num_validation_points;
    get_range_edge_points(P, idx_edge, &n, x, q, &(cu_model->range_edges_model_views), &(cu_model->range_edges_view_cnt), &(cu_model->range_edges_view_idx), &(cu_model->range_edges_points), &r[2]);
    cu_transform_cloud(P, P, n, x, q);
    /*
    if (cu_params->num_validation_points == 0) {
      int num_occ_edges;
      compute_occ_edges(occ_edges, V_edge, V2_edge, &num_occ_edges, xi, yi, vis_prob, num_validation_points, &(cu_obs->range_image_data), cu_params);
      edge_score = compute_edge_score(P, vis_prob_edge, vis_pmf_edge, n, occ_edges, num_occ_edges, &(cu_obs->range_image_data), &(cu_obs->range_image), &(cu_obs->edge_image), cu_params, score_round);
    }
    else*/
    edge_score = compute_edge_score(P, vis_prob_edge, vis_pmf_edge, n, NULL, 0, &(cu_obs->range_image_data), &(cu_obs->range_image), &(cu_obs->edge_image), cu_params, score_round);
  }

  /*if (dbg_timed) {
    printf("break 6, %.2f ms\n", get_time_ms() - t0);  //dbug
    t0 = get_time_ms();
    }*/

  /*double segment_score = 0;
  if (score_round >= 3)
    segment_score = compute_segment_score(x, q, cloud, model_data->model_xyz_index, &model_data->model_xyz_params, vis_prob,
					  num_validation_points, obs_data->obs_range_image, obs_data->obs_edge_image, params, score_round);
  */

  // (Sanja) double score = xyz_score + normal_score + edge_score + lab_score + vis_score + segment_score + fpfh_score + labdist_score;
  double score = xyz_score + normal_score + lab_score + vis_score + edge_score;

  //dbug
  //if (sample->c_type[0] == C_TYPE_SIFT)
  //  score += 100;

  /*if (dbg_timed) {
    printf("break 7, %.2f ms\n", get_time_ms() - t0);  //dbug
    } */
    
  return score;
}

__global__ void score_samples(double *cu_scores, cu_double_matrix_t cu_samples_x, cu_double_matrix_t cu_samples_q, int num_samples, cu_model_data_t cu_model, cu_obs_data_t cu_obs, scope_params_t cu_params, 
			      int score_round, int *cu_xi, int *cu_yi, int *cu_idx, double *cu_cloud, double *cu_cloud_normals, double *cu_cloud_lab, double *cu_vis_prob, double *cu_vis_pmf, uint *cu_rands, 
			      scope_noise_model_t *cu_noise_models,
			      int *cu_idx_edge, double *cu_P, double *cu_V_edge, double *cu_V2_edge, int *cu_occ_edge, double *cu_vis_prob_edge, double *cu_vis_pmf_edge) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_samples) {
    //printf("*****************gpu sample %d\n", i);
    int n_edge = cu_model.max_num_edges;
    double x[3], q[4];
    int j;
    double *row = get_row(&cu_samples_x, i);
    for (j = 0; j < 3; ++j)
      x[j] = row[j];
    row = get_row(&cu_samples_q, i);
    for (j = 0; j < 4; ++j)
      q[j] = row[j];
    //printf("%d\n", cu_params.num_validation_points);
    //printf("%d\n", cu_model.num_points);
    int num_validation_points = (cu_params.num_validation_points > 0 ? cu_params.num_validation_points : cu_model.num_points);
    int *xi, *yi, *idx;
    double *cloud, *cloud_normals, *cloud_lab;
    double *vis_pmf, *vis_prob;
    scope_noise_model_t *noise_models;
    xi = &cu_xi[i*num_validation_points];
    yi = &cu_yi[i*num_validation_points];
    idx = &cu_idx[i*num_validation_points];
    cloud = &cu_cloud[3*i*num_validation_points];
    cloud_normals = &cu_cloud_normals[3*i*num_validation_points];
    cloud_lab = &cu_cloud_lab[3*i*num_validation_points];
    vis_prob = &cu_vis_prob[i*num_validation_points];
    vis_pmf = &cu_vis_pmf[i*num_validation_points];
    noise_models = &cu_noise_models[i*num_validation_points];

    int w, h;
    w = cu_obs.range_image_data.w;
    h = cu_obs.range_image_data.h;

    int *idx_edge, *occ_edge; 
    double *P, *V_edge, *V2_edge, *vis_prob_edge, *vis_pmf_edge;
    idx_edge = &cu_idx_edge[i*n_edge];
    P = &cu_P[i*n_edge*3];
    V_edge = &cu_V_edge[i*w*h];
    V2_edge = &cu_V2_edge[i*w*h];
    occ_edge = &cu_occ_edge[i*w*h*2];
    vis_prob_edge = &cu_vis_prob_edge[i*n_edge];
    vis_pmf_edge = &cu_vis_pmf_edge[i*n_edge];

    cu_scores[i] = cu_model_placement_score(x, q, &cu_model, &cu_obs, &cu_params, score_round, xi, yi, idx, cloud, cloud_normals, cloud_lab, num_validation_points, 
					    vis_prob, vis_pmf, &cu_rands[4*i], noise_models,
					    idx_edge, P, V_edge, V2_edge, occ_edge, vis_prob_edge, vis_pmf_edge);

    //printf("%lf\n", cu_scores[i]);
  }
}

void copy_double_matrix_to_gpu(cu_double_matrix_t *dev_dest, double **host_src, int n, int m) {
  dev_dest->n = n;
  dev_dest->m = m;
  //cudaMallocPitch(&(dev_dest->ptr), &(dev_dest->pitch), m * sizeof(double), n);
  //cudaMemcpy2D(dev_dest->ptr, dev_dest->pitch, host_src[0], m * sizeof(double), m * sizeof(double), n, cudaMemcpyHostToDevice); 
  if (cudaMalloc(&(dev_dest->ptr), m*n*sizeof(double)) != cudaSuccess) {
    printf("double 2d malloc\n");
  }      
  if (cudaMemcpy(dev_dest->ptr, host_src[0], n * m * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("double 2d copy\n");
  }      
} 

void copy_int_matrix_to_gpu(cu_int_matrix_t *dev_dest, int **host_src, int n, int m) {
  dev_dest->n = n;
  dev_dest->m = m;
  //cudaMallocPitch(&(dev_dest->ptr), &(dev_dest->pitch), m * sizeof(int), n);
  //cudaMemcpy2D(dev_dest->ptr, dev_dest->pitch, host_src[0], m * sizeof(int), m * sizeof(int), n, cudaMemcpyHostToDevice); 
  if (cudaMalloc(&(dev_dest->ptr), m*n*sizeof(int)) != cudaSuccess) {
    printf("int 2d malloc \n");
  }      
  if (cudaMemcpy(dev_dest->ptr, host_src[0], n * m * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("int 2d copy\n");
  }      
}

void copy_double_matrix3d_to_gpu(cu_double_matrix3d_t *dev_dest, double ***host_src, int n, int m, int p) {
  dev_dest->n = n; dev_dest->m = m; dev_dest->p = p;
  /*  dest->extent = make_cudaExtent(m * sizeof(double), n, p);
  cudaMalloc3D(&(dest->ptr), dest->extent);
  cudaPitchedPtr src_ptr;
  src_ptr.ptr = host_src[0][0];
  src_ptr.pitch = m * sizeof(double);
  src_ptr.xsize = m;
  src_ptr.ysize = n;
  cudaMemcpy3DParms copy_params = {0};
  copy_params.srcPtr = src_ptr;
  copy_params.dstPtr = dest->ptr;
  copy_params.extent = dest->extent;
  copy_params.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copy_params);*/
  if (cudaMalloc(&(dev_dest->ptr), n * m * p * sizeof(double)) != cudaSuccess) {
    printf("3d malloc\n");
  }      
  if (cudaMemcpy(dev_dest->ptr, host_src[0][0], n * m * p * sizeof(double), cudaMemcpyHostToDevice)) {
    printf("3d copy\n");
  }
}

void copy_double_arr_to_gpu(cu_double_arr_t *dev_dest, double *host_src, int n) {
  dev_dest->n = n;
  if (cudaMalloc(&(dev_dest->ptr), n * sizeof(double)) != cudaSuccess) {
    printf("double arr malloc\n");
  }
  if (cudaMemcpy(dev_dest->ptr, host_src, n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("double arr copy\n");
  }
}

void copy_int_arr_to_gpu(cu_int_arr_t *dev_dest, int *host_src, int n) {
  dev_dest->n = n;
  if (cudaMalloc(&(dev_dest->ptr), n * sizeof(int)) != cudaSuccess) {
    printf("int arr malloc\n");
  }
  if (cudaMemcpy(dev_dest->ptr, host_src, n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("int arr copy\n");
  }
}

void cu_score_samples(double *scores, scope_sample_t *samples, int num_samples, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, scope_params_t *cu_params, int score_round, int num_validation_points) {
  double t0 = get_time_ms();  
  int *cu_xi, *cu_yi, *cu_idx;
  double *cu_cloud, *cu_normals, *cu_lab;
  double *cu_vis_prob, *cu_vis_pmf;
  scope_noise_model_t *cu_noise_models;
  
  // NOTE(sanja): possible perf optimization: do all these in one giant Malloc. Downside: it might be hard to find this big chunk of memory. Solution: chunk it up a little bit.
  if (cudaMalloc(&cu_xi, num_samples * num_validation_points * sizeof(int)) != cudaSuccess) {
    printf("xi malloc\n");
  }      
  if (cudaMalloc(&cu_yi, num_samples * num_validation_points * sizeof(int)) != cudaSuccess) {
    printf("yi\n");
  }      
  if (cudaMalloc(&cu_idx, num_samples * num_validation_points * sizeof(int)) != cudaSuccess) {
    printf("idx\n");
  }      
  if (cudaMalloc(&cu_cloud, num_samples * num_validation_points * 3 * sizeof(double)) != cudaSuccess) {
    printf("cloud\n");
  }
  if (cudaMalloc(&cu_lab, num_samples * num_validation_points * 3 * sizeof(double)) != cudaSuccess) {
    printf("lab\n");
  }      
  if (cudaMalloc(&cu_normals, num_samples * num_validation_points * 3 * sizeof(double)) != cudaSuccess) {
    printf("normals\n");
  }      
  if (cudaMalloc(&cu_vis_prob, num_samples * num_validation_points * sizeof(double)) != cudaSuccess) {
    printf("vis_prob\n");
  }      
  if (cudaMalloc(&cu_vis_pmf, num_samples * num_validation_points * sizeof(double)) != cudaSuccess) {
    printf("vis_pmf\n");
  }      
  if (cudaMalloc(&cu_noise_models, num_samples * num_validation_points * sizeof(scope_noise_model_t)) != cudaSuccess) {
    printf("noise_models\n");
  }      
  
  // edge stuff
  int *cu_idx_edge, *cu_occ_edges;
  double *cu_P, *cu_V_edge, *cu_V2_edge, *cu_vis_prob_edge, *cu_vis_pmf_edge;
  int n_edge = cu_model->max_num_edges;
  if (cudaMalloc(&cu_idx_edge, num_samples * n_edge * sizeof(int)) != cudaSuccess) {
    printf("idx_edge\n");
  }
  if (cudaMalloc(&cu_P, num_samples * n_edge * 3 * sizeof(double)) != cudaSuccess) {
    printf("P\n");
  }
  int w, h;
  w = cu_obs->range_image_data.w;
  h = cu_obs->range_image_data.h;
  if (cudaMalloc(&cu_V_edge, num_samples * w * h * sizeof(double)) != cudaSuccess) {
    printf("V_edge\n");
  }
  if (cudaMalloc(&cu_V2_edge, num_samples * w * h * sizeof(double)) != cudaSuccess) {
    printf("V2_edge\n");
  }
  if (cudaMalloc(&cu_occ_edges, num_samples * w * h * 2 * sizeof(int)) != cudaSuccess) {
    printf("occ_edges\n");
  }
  if (cudaMalloc(&cu_vis_prob_edge, num_samples * n_edge * sizeof(double)) != cudaSuccess) {
    printf("vis_prob_edge\n");
  }
  if (cudaMalloc(&cu_vis_pmf_edge, num_samples * n_edge * sizeof(double)) != cudaSuccess) {
    printf("vis_pmf_edge\n");
  }

  double *cu_scores;
  if (cudaMalloc(&cu_scores, num_samples * sizeof(double)) != cudaSuccess) {
    printf("scores\n");
  }      

  uint *cu_rands;
  if (cudaMalloc(&cu_rands, 2 * num_samples * sizeof(double)) != cudaSuccess) {
    printf("rands\n");
  }      
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  curandGenerate(gen, cu_rands, 4 * num_samples);

  double **samples_x = new_matrix2(num_samples, 3);  
  double **samples_q = new_matrix2(num_samples, 4);
  int i;
  for (i = 0; i < num_samples; ++i) {
    memcpy(samples_x[i], samples[i].x, 3 * sizeof(double));
  }
  // I believe it is cache friendlier to copy like this
  for (i = 0; i < num_samples; ++i) {
    memcpy(samples_q[i], samples[i].q, 4 * sizeof(double));
  }  

  cu_double_matrix_t cu_samples_x;
  copy_double_matrix_to_gpu(&cu_samples_x, samples_x, num_samples, 3);
  cu_double_matrix_t cu_samples_q;
  copy_double_matrix_to_gpu(&cu_samples_q, samples_q, num_samples, 4);
  
  //cudaProfilerStart();
  int threads_per_block = 8;
  int blocks_per_grid = ceil(num_samples/(1.0*threads_per_block));
  score_samples<<<blocks_per_grid, threads_per_block>>>(cu_scores, cu_samples_x, cu_samples_q, num_samples, *cu_model, *cu_obs, *cu_params, score_round, 
							cu_xi, cu_yi, cu_idx, cu_cloud, cu_normals, cu_lab, cu_vis_prob, cu_vis_pmf, cu_rands, cu_noise_models,
							cu_idx_edge, cu_P, cu_V_edge, cu_V2_edge, cu_occ_edges, cu_vis_prob_edge, cu_vis_pmf_edge);
  cudaDeviceSynchronize();
  //cudaProfilerStop();
  printf("scoring: %.2f ms\n", get_time_ms() - t0);
  cudaError err;
  err = cudaMemcpy(scores, cu_scores, num_samples * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("scores error: %s\n", cudaGetErrorString(err));
  }      
  
  free_matrix2(samples_x);
  free_matrix2(samples_q);  

  if (cudaFree(cu_xi)) {
    printf("free xi\n");
  }      
  if (cudaFree(cu_yi)  != cudaSuccess) {
    printf("free yi\n");
  }      
  if (cudaFree(cu_idx) != cudaSuccess) {
    printf("free idx\n");
  }      
  if (cudaFree(cu_cloud) != cudaSuccess) {
    printf("free cloud\n");
  }      
  if (cudaFree(cu_normals) != cudaSuccess) {
    printf("free normals\n");
  }      
  if (cudaFree(cu_lab) != cudaSuccess) {
    printf("free lab\n");
  }
  if (cudaFree(cu_rands) != cudaSuccess) {
    printf("free rands\n");
  }      
  if (cudaFree(cu_vis_pmf) != cudaSuccess) {
    printf("free vis_pmf\n");
  }      
  if (cudaFree(cu_vis_prob) != cudaSuccess) {
    printf("free vis_prob\n");
  }      
  if (cudaFree(cu_noise_models) != cudaSuccess) {
    printf("free noise_models\n");
  }
  if (cudaFree(cu_idx_edge) != cudaSuccess) {
    printf("free idx_edge\n");
  }
  cu_free(cu_P, "free P\n");
  cu_free(cu_V_edge, "free V\n");
  cu_free(cu_V2_edge, "free V2\n");
  cu_free(cu_occ_edges, "free occ_edges\n");
  cu_free(cu_vis_prob_edge, "free_vis_prob_edge\n");
  cu_free(cu_vis_pmf_edge, "free vis_pmf_edge\n");
}

void cu_init() {
  CUresult err = cuInit(0);
  //if (err != 0) 
  printf("Init error: %d\n", err);
}

void cu_init_scoring(scope_model_data_t *model_data, scope_obs_data_t *obs_data, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs) {

  // Allocate all the memory
  copy_double_matrix_to_gpu(&(cu_model->points), model_data->pcd_model->points, model_data->pcd_model->num_points, 3);
  copy_double_matrix_to_gpu(&(cu_model->normals), model_data->pcd_model->normals, model_data->pcd_model->num_points, 3);
  copy_double_arr_to_gpu(&(cu_model->normalvar), model_data->pcd_model->normalvar, model_data->pcd_model->num_points);
  copy_double_matrix_to_gpu(&(cu_model->lab), model_data->pcd_model->lab, model_data->pcd_model->num_points, 3);
  copy_double_matrix_to_gpu(&(cu_model->ved), model_data->pcd_model->ved, model_data->pcd_model->num_points, 66);
  copy_double_matrix_to_gpu(&(cu_model->color_avg_cov), model_data->color_model->avg_cov, 3, 3);
  copy_int_arr_to_gpu(&(cu_model->color_cnts1), model_data->color_model->cnts[0], model_data->color_model->num_points);
  copy_int_arr_to_gpu(&(cu_model->color_cnts2), model_data->color_model->cnts[1], model_data->color_model->num_points);
  copy_double_matrix_to_gpu(&(cu_model->color_means1), model_data->color_model->means[0], model_data->color_model->num_points, 3);
  copy_double_matrix_to_gpu(&(cu_model->color_means2), model_data->color_model->means[1], model_data->color_model->num_points, 3);
  copy_double_matrix3d_to_gpu(&(cu_model->color_cov1), model_data->color_model->covs[0], model_data->color_model->num_points, 3, 3);
  copy_double_matrix3d_to_gpu(&(cu_model->color_cov2), model_data->color_model->covs[1], model_data->color_model->num_points, 3, 3);
  //copy_double_matrix_to_gpu(&(cu_model->fpfh_shapes), model_data->fpfh_model->shapes, model_data->fpfh_model->shape_length, 33);
  copy_double_matrix_to_gpu(&(cu_model->range_edges_model_views), model_data->range_edges_model->views, model_data->range_edges_model->num_views, 3);
  copy_int_arr_to_gpu(&(cu_model->range_edges_view_idx), model_data->range_edges_model->view_idx, model_data->range_edges_model->num_views);
  copy_int_arr_to_gpu(&(cu_model->range_edges_view_cnt), model_data->range_edges_model->view_cnt, model_data->range_edges_model->num_views);
  copy_double_matrix_to_gpu(&(cu_model->range_edges_points), model_data->range_edges_model->pcd->points, model_data->range_edges_model->pcd->num_points, 3);
  
  cu_model->num_points = model_data->pcd_model->num_points;
  cu_model->num_views = model_data->range_edges_model->num_views;
  int n_edge = arr_max_i(model_data->range_edges_model->view_cnt, model_data->range_edges_model->num_views);
  cu_model->max_num_edges = n_edge;
  // CONTINUE HERE FOR MODEL DATA COPYING ****************************

  copy_double_matrix_to_gpu(&(cu_obs->range_image), obs_data->obs_range_image->image, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_int_matrix_to_gpu(&(cu_obs->range_image_cnt), obs_data->obs_range_image->cnt, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_double_matrix3d_to_gpu(&(cu_obs->range_image_points), obs_data->obs_range_image->points, obs_data->obs_range_image->w, obs_data->obs_range_image->h, 3);
  copy_double_matrix3d_to_gpu(&(cu_obs->range_image_normals), obs_data->obs_range_image->normals, obs_data->obs_range_image->w, obs_data->obs_range_image->h, 3);
  copy_int_matrix_to_gpu(&(cu_obs->range_image_idx), obs_data->obs_range_image->idx, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_double_matrix_to_gpu(&(cu_obs->range_image_pcd_obs_bg_lab), obs_data->pcd_obs_bg->lab, obs_data->pcd_obs_bg->num_points, 3);
  //copy_double_matrix_to_gpu(&(cu_obs->pcd_obs_fpfh), obs_data->pcd_obs->fpfh, obs_data->pcd_obs->fpfh_length, 33);
  copy_double_matrix_to_gpu(&(cu_obs->edge_image), obs_data->obs_edge_image, obs_data->obs_range_image->w, obs_data->obs_range_image->h);

  cu_obs->range_image_data.res = obs_data->obs_range_image->res;
  cu_obs->range_image_data.min0 = obs_data->obs_range_image->min[0];
  cu_obs->range_image_data.min1 = obs_data->obs_range_image->min[1];
  cu_obs->range_image_data.w = obs_data->obs_range_image->w;
  cu_obs->range_image_data.h = obs_data->obs_range_image->h;

  // CONTINUE HERE FOR OBS DATA COPYING ********************************

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  //cudaDeviceSynchronize();
}

void cu_init_scoring_mope(scope_model_data_t model_data[], scope_obs_data_t *obs_data, int num_models, cu_model_data_t cu_model[], cu_obs_data_t *cu_obs) {
  
  // Allocate all the memory

  for (int i = 0; i < num_models; ++i) {
    copy_double_matrix_to_gpu(&(cu_model[i].points), model_data[i].pcd_model->points, model_data[i].pcd_model->num_points, 3);
    copy_double_matrix_to_gpu(&(cu_model[i].normals), model_data[i].pcd_model->normals, model_data[i].pcd_model->num_points, 3);
    copy_double_arr_to_gpu(&(cu_model[i].normalvar), model_data[i].pcd_model->normalvar, model_data[i].pcd_model->num_points);
    copy_double_matrix_to_gpu(&(cu_model[i].lab), model_data[i].pcd_model->lab, model_data[i].pcd_model->num_points, 3);
    copy_double_matrix_to_gpu(&(cu_model[i].ved), model_data[i].pcd_model->ved, model_data[i].pcd_model->num_points, 66);
    copy_double_matrix_to_gpu(&(cu_model[i].color_avg_cov), model_data[i].color_model->avg_cov, 3, 3);
    copy_int_arr_to_gpu(&(cu_model[i].color_cnts1), model_data[i].color_model->cnts[0], model_data[i].color_model->num_points);
    copy_int_arr_to_gpu(&(cu_model[i].color_cnts2), model_data[i].color_model->cnts[1], model_data[i].color_model->num_points);
    copy_double_matrix_to_gpu(&(cu_model[i].color_means1), model_data[i].color_model->means[0], model_data[i].color_model->num_points, 3);
    copy_double_matrix_to_gpu(&(cu_model[i].color_means2), model_data[i].color_model->means[1], model_data[i].color_model->num_points, 3);
    copy_double_matrix3d_to_gpu(&(cu_model[i].color_cov1), model_data[i].color_model->covs[0], model_data[i].color_model->num_points, 3, 3);
    copy_double_matrix3d_to_gpu(&(cu_model[i].color_cov2), model_data[i].color_model->covs[1], model_data[i].color_model->num_points, 3, 3);
    //copy_double_matrix_to_gpu(&(cu_model->fpfh_shapes), model_data->fpfh_model->shapes, model_data->fpfh_model->shape_length, 33);
    copy_double_matrix_to_gpu(&(cu_model[i].range_edges_model_views), model_data[i].range_edges_model->views, model_data[i].range_edges_model->num_views, 3);
    copy_int_arr_to_gpu(&(cu_model[i].range_edges_view_idx), model_data[i].range_edges_model->view_idx, model_data[i].range_edges_model->num_views);
    copy_int_arr_to_gpu(&(cu_model[i].range_edges_view_cnt), model_data[i].range_edges_model->view_cnt, model_data[i].range_edges_model->num_views);
    copy_double_matrix_to_gpu(&(cu_model[i].range_edges_points), model_data[i].range_edges_model->pcd->points, model_data[i].range_edges_model->pcd->num_points, 3);
  
    cu_model[i].num_points = model_data[i].pcd_model->num_points;
    cu_model[i].num_views = model_data[i].range_edges_model->num_views;
    int n_edge = arr_max_i(model_data[i].range_edges_model->view_cnt, model_data[i].range_edges_model->num_views);
    cu_model[i].max_num_edges = n_edge;
    // CONTINUE HERE FOR MODEL DATA COPYING ****************************
  }

  copy_double_matrix_to_gpu(&(cu_obs->range_image), obs_data->obs_range_image->image, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_int_matrix_to_gpu(&(cu_obs->range_image_cnt), obs_data->obs_range_image->cnt, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_double_matrix3d_to_gpu(&(cu_obs->range_image_points), obs_data->obs_range_image->points, obs_data->obs_range_image->w, obs_data->obs_range_image->h, 3);
  copy_double_matrix3d_to_gpu(&(cu_obs->range_image_normals), obs_data->obs_range_image->normals, obs_data->obs_range_image->w, obs_data->obs_range_image->h, 3);
  copy_int_matrix_to_gpu(&(cu_obs->range_image_idx), obs_data->obs_range_image->idx, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_double_matrix_to_gpu(&(cu_obs->range_image_pcd_obs_bg_lab), obs_data->pcd_obs_bg->lab, obs_data->pcd_obs_bg->num_points, 3);
  //copy_double_matrix_to_gpu(&(cu_obs->pcd_obs_fpfh), obs_data->pcd_obs->fpfh, obs_data->pcd_obs->fpfh_length, 33);
  copy_double_matrix_to_gpu(&(cu_obs->edge_image), obs_data->obs_edge_image, obs_data->obs_range_image->w, obs_data->obs_range_image->h);

  cu_obs->range_image_data.res = obs_data->obs_range_image->res;
  cu_obs->range_image_data.min0 = obs_data->obs_range_image->min[0];
  cu_obs->range_image_data.min1 = obs_data->obs_range_image->min[1];
  cu_obs->range_image_data.w = obs_data->obs_range_image->w;
  cu_obs->range_image_data.h = obs_data->obs_range_image->h;

  // CONTINUE HERE FOR OBS DATA COPYING ********************************

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  //cudaDeviceSynchronize();
}

void cu_free_all_the_things(cu_model_data_t *cu_model, cu_obs_data_t *cu_obs) {
  // Free ALL the things!!!

  cudaFree(cu_model->points.ptr);
  cudaFree(cu_model->normals.ptr);
  cudaFree(cu_model->normalvar.ptr);
  cudaFree(cu_model->lab.ptr);
  cudaFree(cu_model->ved.ptr);
  cudaFree(cu_model->color_avg_cov.ptr);
  cudaFree(cu_model->color_means1.ptr);
  cudaFree(cu_model->color_means2.ptr);
  //cudaFree(cu_model->fpfh_shapes.ptr);
  cudaFree(cu_model->range_edges_model_views.ptr);
  cudaFree(cu_model->range_edges_points.ptr);
  cudaFree(cu_model->color_cov1.ptr);
  cudaFree(cu_model->color_cov2.ptr);
  cudaFree(cu_model->color_cnts1.ptr);
  cudaFree(cu_model->color_cnts2.ptr);
  cudaFree(cu_model->range_edges_view_idx.ptr);
  cudaFree(cu_model->range_edges_view_cnt.ptr);
  
  cudaFree(cu_obs->range_image.ptr);
  cudaFree(cu_obs->range_image_idx.ptr);
  cudaFree(cu_obs->range_image_pcd_obs_bg_lab.ptr);
  //cudaFree(cu_obs->pcd_obs_fpfh.ptr);
  cudaFree(cu_obs->edge_image.ptr);
  cudaFree(cu_obs->range_image_points.ptr);
  cudaFree(cu_obs->range_image_normals.ptr);
  cudaFree(cu_obs->range_image_cnt.ptr);
 
  curandDestroyGenerator(gen);
}

void cu_free_all_the_things_mope(cu_model_data_t cu_model[], cu_obs_data_t *cu_obs, int num_models) {
  // Free ALL the things!!!
  
  for (int i = 0; i < num_models; ++i) {
    cudaFree(cu_model[i].points.ptr);
    cudaFree(cu_model[i].normals.ptr);
    cudaFree(cu_model[i].normalvar.ptr);
    cudaFree(cu_model[i].lab.ptr);
    cudaFree(cu_model[i].ved.ptr);
    cudaFree(cu_model[i].color_avg_cov.ptr);
    cudaFree(cu_model[i].color_means1.ptr);
    cudaFree(cu_model[i].color_means2.ptr);
    //cudaFree(cu_model->fpfh_shapes.ptr);
    cudaFree(cu_model[i].range_edges_model_views.ptr);
    cudaFree(cu_model[i].range_edges_points.ptr);
    cudaFree(cu_model[i].color_cov1.ptr);
    cudaFree(cu_model[i].color_cov2.ptr);
    cudaFree(cu_model[i].color_cnts1.ptr);
    cudaFree(cu_model[i].color_cnts2.ptr);
    cudaFree(cu_model[i].range_edges_view_idx.ptr);
    cudaFree(cu_model[i].range_edges_view_cnt.ptr);
  }
  cudaFree(cu_obs->range_image.ptr);
  cudaFree(cu_obs->range_image_idx.ptr);
  cudaFree(cu_obs->range_image_pcd_obs_bg_lab.ptr);
  //cudaFree(cu_obs->pcd_obs_fpfh.ptr);
  cudaFree(cu_obs->edge_image.ptr);
  cudaFree(cu_obs->range_image_points.ptr);
  cudaFree(cu_obs->range_image_normals.ptr);
  cudaFree(cu_obs->range_image_cnt.ptr);
 
  curandDestroyGenerator(gen);
}

