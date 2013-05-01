#include "cuda.h"
#include "include/bingham/cuda_wrapper.h"
#include "curand.h"
#include "bingham/olf.h"

#include <math.h>

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

#define cu_malloc(x, sz, msg) do{ if (cudaMalloc(x, sz) != cudaSuccess) printf(msg); } while (0)
#define cu_free(x, msg) do{ if (cudaFree(x) != cudaSuccess) printf(msg); } while (0)

curandGenerator_t gen;

//#define CUDA_LAUNCH_BLOCKING 1

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

__device__ inline double cu_sigmoid(double x, const double *b)
{
  return b[0] + (1 - b[0]) / (1 + exp(-b[1]-b[2]*x));
}

__device__ inline double cu_logistic(double x, double *b)
{
  return 1.0 / (1.0 + exp(-x*b[1]-b[0]));
}

void copy_double_matrix_to_gpu(cu_double_matrix_t *dev_dest, double **host_src, int n, int m) {
  dev_dest->n = n;
  dev_dest->m = m;
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
  if (cudaMalloc(&(dev_dest->ptr), m*n*sizeof(int)) != cudaSuccess) {
    printf("int 2d malloc \n");
  }      
  if (cudaMemcpy(dev_dest->ptr, host_src[0], n * m * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("int 2d copy\n");
  }      
}

void copy_double_matrix3d_to_gpu(cu_double_matrix3d_t *dev_dest, double ***host_src, int n, int m, int p) {
  dev_dest->n = n; dev_dest->m = m; dev_dest->p = p;
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

__device__ double cu_dot(double x[], double y[], int n) {
  int i;
  double z = 0.0;
  for (i = 0; i < n; i++)
    z += x[i]*y[i];
  return z;
}

__device__ void cu_matrix_vec_mult_3(double *y, double A[][3], double *x, int n) {
  int i;
  if (y == x) { // dbug
    printf("**************FIX CU_MATRIX_VEC_MULT CALL!\n");
  }
  for (i = 0; i < n; i++)
    y[i] = cu_dot(A[i], x, 3);
}

// matrix multiplication, Z = X*Y, where X is n-by-p and Y is p-by-m
__device__ void cu_matrix_mult_2_3_4(double Z[][4], double X[][3], double Y[][4])
{
  int i, j, k;
  for (i = 0; i < 2; i++) {     // row i
    for (j = 0; j < 4; j++) {   // column j
      Z[i][j] = 0;
      for (k = 0; k < 3; k++)
	Z[i][j] += X[i][k]*Y[k][j];
    }
  }
}

__device__ void cu_vec_matrix_mult_7(double y[], double x[], double A[][7], int n)
{
  int i, j;
  if (y == x) {
    printf("****************FIX vec_matrix_mult call!\n");
  }
  else {
    for (j = 0; j < 7; j++) {
      y[j] = 0;
      for (i = 0; i < n; i++)
	y[j] += x[i]*A[i][j];
    }
  }
}

__device__ void cu_vec_matrix_mult_4(double y[], double x[], double A[][4], int n)
{
  int i, j;
  if (y == x) {
    printf("****************FIX vec_matrix_mult call!\n");
  }
  else {
    for (j = 0; j < 4; j++) {
      y[j] = 0;
      for (i = 0; i < n; i++)
	y[j] += x[i]*A[i][j];
    }
  }
}

// adds two vectors, z = x+y
__device__ void cu_add(double z[], double x[], double y[], int n) {
  int i;
  for (i = 0; i < n; i++)
    z[i] = x[i] + y[i];
}

__device__ double cu_norm(double x[], int n) {
  double d = 0.0;
  int i;

  for (i = 0; i < n; i++)
    d += x[i]*x[i];

  return sqrt(d);
}

__device__ void cu_normalize(double y[], double x[], int n) {
  double d = cu_norm(x, n);
  int i;
  for (i = 0; i < n; i++)
    y[i] = x[i]/d;
}

// compute the pdf of a normal random variable
__device__ double cu_normpdf(double x, double mu, double sigma) {
  double dx = x - mu;

  return exp(-dx*dx / (2*sigma*sigma)) / (sqrt(2*M_PI) * sigma);
}

// invert a quaternion
__device__ void cu_quaternion_inverse(double q_inv[4], double *q) {
  q_inv[0] = q[0];
  q_inv[1] = -q[1];
  q_inv[2] = -q[2];
  q_inv[3] = -q[3];
}

// multiplies a vector by a scalar, y = c*x
__device__ void cu_mult(double y[], double x[], double c, int n)
{
  int i;
  for (i = 0; i < n; i++)
    y[i] = c*x[i];
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

void cu_init() {
  CUresult err = cuInit(0);
  //if (err != 0) 
  printf("Init error: %d\n", err);
}

void cu_init_model(scope_model_data_t *model_data, cu_model_data_t *cu_model) {
  // Allocate all the memory
  copy_double_matrix_to_gpu(&(cu_model->points), model_data->pcd_model->points, model_data->pcd_model->num_points, 3);
  copy_double_matrix_to_gpu(&(cu_model->normals), model_data->pcd_model->normals, model_data->pcd_model->num_points, 3);
  copy_double_arr_to_gpu(&(cu_model->normalvar), model_data->pcd_model->normalvar, model_data->pcd_model->num_points);
  copy_double_matrix_to_gpu(&(cu_model->lab), model_data->pcd_model->lab, model_data->pcd_model->num_points, 3);
  copy_double_matrix_to_gpu(&(cu_model->ved), model_data->pcd_model->ved, model_data->pcd_model->num_points, 66);
  /*copy_double_matrix_to_gpu(&(cu_model->color_avg_cov), model_data->color_model->avg_cov, 3, 3);
  copy_int_arr_to_gpu(&(cu_model->color_cnts1), model_data->color_model->cnts[0], model_data->color_model->num_points);
  copy_int_arr_to_gpu(&(cu_model->color_cnts2), model_data->color_model->cnts[1], model_data->color_model->num_points);
  copy_double_matrix_to_gpu(&(cu_model->color_means1), model_data->color_model->means[0], model_data->color_model->num_points, 3);
  copy_double_matrix_to_gpu(&(cu_model->color_means2), model_data->color_model->means[1], model_data->color_model->num_points, 3);
  copy_double_matrix3d_to_gpu(&(cu_model->color_cov1), model_data->color_model->covs[0], model_data->color_model->num_points, 3, 3);
  copy_double_matrix3d_to_gpu(&(cu_model->color_cov2), model_data->color_model->covs[1], model_data->color_model->num_points, 3, 3);*/
  //copy_double_matrix_to_gpu(&(cu_model->fpfh_shapes), model_data->fpfh_model->shapes, model_data->fpfh_model->shape_length, 33);
  copy_double_matrix_to_gpu(&(cu_model->range_edges_model_views), model_data->range_edges_model->views, model_data->range_edges_model->num_views, 3);
  copy_int_arr_to_gpu(&(cu_model->range_edges_view_idx), model_data->range_edges_model->view_idx, model_data->range_edges_model->num_views);
  copy_int_arr_to_gpu(&(cu_model->range_edges_view_cnt), model_data->range_edges_model->view_cnt, model_data->range_edges_model->num_views);
  copy_double_matrix_to_gpu(&(cu_model->range_edges_points), model_data->range_edges_model->pcd->points, model_data->range_edges_model->pcd->num_points, 3);

  cudaMalloc(&(cu_model->score_comp_models), sizeof(score_comp_models_t));
  cudaMemcpy(cu_model->score_comp_models, model_data->score_comp_models, sizeof(score_comp_models_t), cudaMemcpyHostToDevice);
  //memcpy(&cu_model->score_comp_models, model_data->score_comp_models, sizeof(score_comp_models_t));
    
  cu_model->num_points = model_data->pcd_model->num_points;
  cu_model->num_views = model_data->range_edges_model->num_views;
  int n_edge = arr_max_i(model_data->range_edges_model->view_cnt, model_data->range_edges_model->num_views);
  cu_model->max_num_edges = n_edge;
}

void cu_init_obs(scope_obs_data_t *obs_data, cu_obs_data_t *cu_obs, scope_params_t *params) {

  copy_double_matrix_to_gpu(&(cu_obs->range_image), obs_data->obs_range_image->image, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_int_matrix_to_gpu(&(cu_obs->range_image_cnt), obs_data->obs_range_image->cnt, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_double_matrix3d_to_gpu(&(cu_obs->range_image_points), obs_data->obs_range_image->points, obs_data->obs_range_image->w, obs_data->obs_range_image->h, 3);
  copy_double_matrix3d_to_gpu(&(cu_obs->range_image_normals), obs_data->obs_range_image->normals, obs_data->obs_range_image->w, obs_data->obs_range_image->h, 3);
  if (params->use_colors)
    copy_double_matrix3d_to_gpu(&(cu_obs->obs_lab_image), obs_data->obs_lab_image, 3, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_int_matrix_to_gpu(&(cu_obs->range_image_idx), obs_data->obs_range_image->idx, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_double_matrix_to_gpu(&(cu_obs->range_image_pcd_obs_lab), obs_data->pcd_obs->lab, obs_data->pcd_obs->num_points, 3);
  //copy_double_matrix_to_gpu(&(cu_obs->pcd_obs_fpfh), obs_data->pcd_obs->fpfh, obs_data->pcd_obs->fpfh_length, 33);
  copy_double_matrix_to_gpu(&(cu_obs->edge_image), obs_data->obs_edge_image, obs_data->obs_range_image->w, obs_data->obs_range_image->h);
  copy_double_matrix_to_gpu(&(cu_obs->segment_affinities), obs_data->obs_segment_affinities, obs_data->num_obs_segments, obs_data->num_obs_segments);

  cu_obs->range_image_data.res = obs_data->obs_range_image->res;
  cu_obs->range_image_data.min0 = obs_data->obs_range_image->min[0];
  cu_obs->range_image_data.min1 = obs_data->obs_range_image->min[1];
  cu_obs->range_image_data.w = obs_data->obs_range_image->w;
  cu_obs->range_image_data.h = obs_data->obs_range_image->h;
  cu_obs->num_obs_segments = obs_data->num_obs_segments;

  // CONTINUE HERE FOR OBS DATA COPYING ********************************
}



void cu_free_all_the_model_things(cu_model_data_t *cu_model) {
  cudaFree(cu_model->points.ptr);
  cudaFree(cu_model->normals.ptr);
  cudaFree(cu_model->normalvar.ptr);
  cudaFree(cu_model->lab.ptr);
  cudaFree(cu_model->ved.ptr);
  /*cudaFree(cu_model->color_avg_cov.ptr);
  cudaFree(cu_model->color_means1.ptr);
  cudaFree(cu_model->color_means2.ptr);
  cudaFree(cu_model->color_cov1.ptr);
  cudaFree(cu_model->color_cov2.ptr);
  cudaFree(cu_model->color_cnts1.ptr);
  cudaFree(cu_model->color_cnts2.ptr);*/
  //cudaFree(cu_model->fpfh_shapes.ptr);
  cudaFree(cu_model->range_edges_model_views.ptr);
  cudaFree(cu_model->range_edges_points.ptr);
  cudaFree(cu_model->range_edges_view_idx.ptr);
  cudaFree(cu_model->range_edges_view_cnt.ptr);
  cudaFree(cu_model->score_comp_models);
}

void cu_free_all_the_obs_things(cu_obs_data_t *cu_obs, scope_params_t *params) {
  cudaFree(cu_obs->range_image.ptr);
  cudaFree(cu_obs->range_image_idx.ptr);
  cudaFree(cu_obs->range_image_pcd_obs_lab.ptr);
  //cudaFree(cu_obs->pcd_obs_fpfh.ptr);
  cudaFree(cu_obs->edge_image.ptr);
  cudaFree(cu_obs->range_image_points.ptr);
  cudaFree(cu_obs->range_image_normals.ptr);
  cudaFree(cu_obs->range_image_cnt.ptr);
  if (params->use_colors)
    cudaFree(cu_obs->obs_lab_image.ptr);
  cudaFree(cu_obs->segment_affinities.ptr);
}

void cu_free_all_the_things(cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, scope_params_t *cu_params, scope_params_t *params) {
  // Free ALL the things!!!
  cu_free(cu_params, "params");
  cudaError_t cudaerr = cudaGetLastError();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

  cu_free_all_the_model_things(cu_model);
  cu_free_all_the_obs_things(cu_obs, params);
  curandDestroyGenerator(gen);
}

void cu_free_all_the_things_mope(cu_model_data_t cu_model[], cu_obs_data_t *cu_obs, int num_models, scope_params_t *params) {
  // Free ALL the things!!!
  
  for (int i = 0; i < num_models; ++i) {
    cu_free_all_the_model_things(&cu_model[i]);

  }
  cu_free_all_the_obs_things(cu_obs, params);
  curandDestroyGenerator(gen);
}

void cu_init_scoring(scope_model_data_t *model_data, scope_obs_data_t *obs_data, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, scope_params_t **cu_params, scope_params_t *params) {

  cu_malloc(cu_params, sizeof(scope_params_t), "params");
  cudaMemcpy(*cu_params, params, sizeof(scope_params_t), cudaMemcpyHostToDevice);

  cu_init_model(model_data, cu_model);
  cu_init_obs(obs_data, cu_obs, params);

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
}

void cu_init_scoring_mope(scope_model_data_t model_data[], scope_obs_data_t *obs_data, int num_models, cu_model_data_t cu_model[], cu_obs_data_t *cu_obs, scope_params_t *params) {
  
  // Allocate all the memory
  for (int i = 0; i < num_models; ++i) {
    cu_init_model(&model_data[i], &cu_model[i]);
  }
  cu_init_obs(obs_data, cu_obs, params);
  
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
}

__device__ void cu_range_image_xyz2sub(int *i, int *j, cu_range_image_data_t range_image, double xyz[])
{
  //TODO: use range image viewpoint

  double d = cu_norm(xyz, 3);
  double x = atan2(xyz[0], xyz[2]);
  double y = acos(xyz[1] / d);

  int cx = (int)floor((x - range_image.min0) / range_image.res);
  int cy = (int)floor((y - range_image.min1) / range_image.res);

  *i = cx;
  *j = cy;

  if (!((cx >= 0 && cy>=0) && (cx < range_image.w) && (cy < range_image.h))) {
    *i = -1;
    *j = -1;
  }
}

/*                                                                                                                                                                                                                 
 * compute viewpoint (in model coordinates) for model placement (x,q) assuming observed viewpoint = (0,0,0)
 */
__device__ void cu_model_pose_to_viewpoint(double *vp, double *x, double *q)
{
  double q_inv[4];
  cu_quaternion_inverse(q_inv, q);
  double R_inv[3][3];
  cu_quaternion_to_rotation_matrix(R_inv,q_inv);
  cu_matrix_vec_mult_3(vp, R_inv, x, 3);
  cu_mult(vp, vp, -1, 3);
}

__global__ void cu_add_matrix_rows_slow(double *out_array, double *in_matrix, int n, int m, int *m_arr) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i >= n)
    return;

  int limit = m;
  if (m_arr)
    limit = m_arr[i];
  
  out_array[i] = 0.0;
  for (int j = 0; j < limit; ++j) {
    out_array[i] += in_matrix[j + i * m];
  }
}

__global__ void cu_add_matrix_3d_slow(double *out_array, double *in_matrix, int n, int m, int *m_arr, int p) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i >= n)
    return;

  int limit = m;
  if (m_arr)
    limit = m_arr[i];

  for (int k = 0; k < p; ++k) {
    out_array[i*p + k] = 0.0;    
  }
  
  for (int j = 0; j < limit; ++j) {
    for (int k = 0; k < p; ++k) {
      out_array[i*p + k] += in_matrix[j*p + i * m * p + k];
    }
  }
}

__global__ void cu_add_matrix_rows_medium(double *out_array, double *in_matrix, int n, int m, int *m_arr) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i >= n || j >= m)
    return;

  int limit = m;
  if (m_arr)
    limit = m_arr[i];

  extern __shared__ double tmps[];
  tmps[threadIdx.x] = 0.0;
  for (int k = j; k < limit; k += blockDim.x) {
    tmps[threadIdx.x] += in_matrix[k + i * m];
  }

  __syncthreads();
  
  if (j == 0) {
    out_array[i] = 0.0;
    for (int k = 0; k < blockDim.x; ++k) {
      out_array[i] += tmps[k];
    }
  }
}

__global__ void cu_divide_matrix_with_vector(double *out_matrix, double *in_matrix, double *scaling_array, int n, int m, int *m_arr) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= n || j >= m)
    return;
  if (m_arr && j >= m_arr[i])
    return;  

  out_matrix[j + i * m] = in_matrix[j + i * m] / scaling_array[i];
}
    
__global__ void cu_get_validation_points(int *idx, int total_pts, int needed, int num_samples, uint *rands)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= needed || i >= num_samples)
    return;

  if (needed == total_pts) {  // use all the points
    idx[j + i * needed] = j;
  } else {
    idx[j + i * needed] = ((rands[(i << 1)] % total_pts) + (j * (big_primes[rands[(i << 1) + 1] % 100] % total_pts))) % total_pts;
  }
}

__global__ void cu_get_sub_cloud_at_pose(double *cloud, cu_double_matrix_t points, double *x, double *q, int *idx, int num_samples, int n)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= n || i >= num_samples)
    return;

  int i_arr = j + i * n;

  double R[3][3];
  cu_quaternion_to_rotation_matrix(R, &q[i * 4]);
  double dest[3]; // In local memory so we access global memory less
  dest[0] = points.ptr[idx[i_arr] * points.m]; 
  dest[1] = points.ptr[idx[i_arr] * points.m + 1]; 
  dest[2] = points.ptr[idx[i_arr] * points.m + 2];
  double tmp[3];
  cu_matrix_vec_mult_3(tmp, R, dest, 3);
  cu_add(dest, tmp, &x[i * 3], 3);
  cloud[3 * i_arr] = dest[0]; cloud[3*i_arr + 1] = dest[1]; cloud[3*i_arr + 2] = dest[2];
}

__global__ void cu_get_sub_cloud_normals_rotated(double *cloud_normals, cu_double_matrix_t normals, double *q, int *idx, int num_samples, int n)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= n || i >= num_samples)
    return;

  int i_arr = j + i * n;

  double R[3][3];
  cu_quaternion_to_rotation_matrix(R, &q[i * 4]);
  double *row;
  double dest[3];

  row = &normals.ptr[idx[i_arr] * normals.m];

  double tmp[3];
  tmp[0] = row[0]; tmp[1] = row[1]; tmp[2] = row[2];
  cu_matrix_vec_mult_3(dest, R, tmp, 3);
  cloud_normals[3*i_arr] = dest[0]; cloud_normals[3*i_arr+1] = dest[1]; cloud_normals[3*i_arr + 2] = dest[2];
}
__global__ void cu_populate_xi_yi(int *xi, int *yi, double *cloud, cu_range_image_data_t range_image_data, int num_samples, int n, int *n_arr) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (i >= num_samples || j >= n)
    return;
  if (n_arr && j >= n_arr[i])
    return;

  int i_arr = j + i * n;
 
  double dest[3];
  dest[0] = cloud[3*i_arr]; 
  dest[1] = cloud[3*i_arr + 1]; 
  dest[2] = cloud[3*i_arr + 2];
  cu_range_image_xyz2sub(&xi[i_arr], &yi[i_arr], range_image_data, dest);
  if (0)
    printf("%d %d %d\n", i_arr, xi[i_arr], yi[i_arr]);
}

__global__ void cu_compute_visibility_prob(double *cu_vis_prob, double *cu_cloud, double *cu_normals, int *cu_xi, int *cu_yi, cu_range_image_data_t ri_data, 
					   cu_double_matrix_t range_image, double vis_thresh, int search_radius, int num_samples, int n, int *n_arr) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= n)
    return;
  if (n_arr && j >= n_arr[i])
    return;

  int i_arr = j + i * n;

  int xi = cu_xi[i_arr];
  int yi = cu_yi[i_arr];

  double V[3];
  double pt[3];
  pt[0] = cu_cloud[3*i_arr]; pt[1] = cu_cloud[3*i_arr + 1]; pt[2] = cu_cloud[3*i_arr + 2];
  cu_normalize(V, pt, 3);

  if (cu_normals != NULL && cu_dot(V, &cu_normals[3*i_arr], 3) >= -.1) {  // normals pointing away
    cu_vis_prob[i_arr] = 0.0;
    return;
  }


  if (xi == -1 && yi == -1) {
    cu_vis_prob[i_arr] = 0.0;
    return;
  }

  double model_range = cu_norm(pt, 3);
  double obs_range = range_image.ptr[xi * range_image.m + yi];

  if (search_radius > 0) {
    int x0 = MAX(xi - search_radius, 0);
    int x1 = MIN(xi + search_radius, ri_data.w - 1);
    int y0 = MAX(yi - search_radius, 0);
    int y1 = MIN(yi + search_radius, ri_data.h - 1);
    int x, y;
    for (x = x0; x <= x1; x++)
      for (y = y0; y <= y1; y++)
	obs_range = MAX(obs_range, range_image.ptr[x * range_image.m + y]);
  }

  double dR = model_range - obs_range;
  cu_vis_prob[i_arr] = (dR < 0 ? 1.0 : cu_normpdf(dR/vis_thresh, 0, 1) / .3989);  // .3989 = normpdf(0,0,1)
}

__global__ void cu_get_viewpoints(int *vi, int num_samples, double *samples_x, double *samples_q, cu_double_matrix_t range_edges_model_views) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  double vp[3];
  
  cu_model_pose_to_viewpoint(vp, &samples_x[3*i], &samples_q[4*i]);
  double vi_max = -(1<<19);

  int j;
  for (j = 0; j < range_edges_model_views.n; ++j) {
    double tmp = cu_dot(&range_edges_model_views.ptr[j * range_edges_model_views.m], vp, 3);
    if (tmp > vi_max) {
      vi[i] = j;
      vi_max = tmp;
    }
  }
}

__global__ void cu_get_noise_models(scope_noise_model_t *noise_models, double *cloud, double *normals, int *idx, int *vi, cu_double_matrix_t ved, cu_double_arr_t normalvar, int num_samples, int n) {

  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= n || i >= num_samples)
    return;

  // prep for lookup edge distances for closest model viewpoint                                                                                                                                                    

  double surface_angles, edge_dists;
  // compute sigmas                 
                           
  int i_arr = i * n + j;
                                                                                                                                                    
  double normalized[3];
  cu_normalize(normalized, &cloud[3*i_arr], 3);
  surface_angles = 1 + cu_dot(normalized, &normals[3 * i_arr], 3);
  edge_dists = ved.ptr[idx[i_arr] * ved.m + vi[i]];
  noise_models[i_arr].range_sigma = .5*cu_sigmoid(surface_angles, b_SR) + .5*cu_sigmoid(edge_dists, b_ER);
  noise_models[i_arr].normal_sigma = .5*cu_sigmoid(surface_angles, b_SN) + .5*cu_sigmoid(edge_dists, b_EN);
  noise_models[i_arr].lab_sigma[0] = .5*cu_sigmoid(surface_angles, b_SL) + .5*cu_sigmoid(edge_dists, b_EL);
  noise_models[i_arr].lab_sigma[1] = .5*cu_sigmoid(surface_angles, b_SA) + .5*cu_sigmoid(edge_dists, b_EA);
  noise_models[i_arr].lab_sigma[2] = .5*cu_sigmoid(surface_angles, b_SB) + .5*cu_sigmoid(edge_dists, b_EB);
  
  noise_models[i_arr].normal_sigma = MAX(noise_models[i_arr].normal_sigma, normalvar.ptr[idx[i_arr]]);
}

__global__ void cu_transform_cloud(double *cloud2, double *cloud, double *x, double *q, int num_samples, int n, int *n_arr)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= n)
    return;
  if (n_arr && j >= n_arr[i])
    return;

  int i_arr = j + i * n;

  double R[3][3];
  cu_quaternion_to_rotation_matrix(R,&q[4*i]);
 
  double tmp[3];
  cu_matrix_vec_mult_3(tmp, R, &cloud[i_arr*3], 3);
  cloud2[3*i_arr] = tmp[0];
  cloud2[3*i_arr+1] = tmp[1];
  cloud2[3*i_arr+2] = tmp[2];
  if (x != NULL) {
    cu_add(&cloud2[i_arr*3], &cloud2[i_arr*3], &x[3*i], 3);
  }
}

__global__ void cu_reorder_rows(double *cloud, cu_double_matrix_t points, int *idx, int n, int m, int p, int *m_arr) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= n || j >= m)
    return;
  if (m_arr && j >= m_arr[i])
    return;
  
  cloud[i * m * p + j * p] = points.ptr[idx[j + i * m] * p];  
  cloud[i * m * p + j * p + 1] = points.ptr[idx[j + i * m] * p + 1];
  cloud[i * m * p + j * p + 2] = points.ptr[idx[j + i * m] * p + 2];

}

__global__ void cu_compute_xyz_score_individual(double *xyz_score, double *cloud, int *xi, int *yi, double *vis_pmf, scope_noise_model_t *noise_models, int num_samples, int num_validation_points, 
						cu_double_matrix_t range_image, cu_range_image_data_t range_image_data, cu_int_matrix_t range_image_cnt, scope_params_t *params)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= num_validation_points || i >= num_samples)
    return;

  int xyz_score_window = params->xyz_score_window;
  
  int i_arr = j + i * num_validation_points;

  xyz_score[i_arr] = 0.0;

  if (vis_pmf[i_arr] > .01/(double)num_validation_points) {
    double range_sigma = params->range_sigma * noise_models[i_arr].range_sigma;
    double model_range = cu_norm(&cloud[3*i_arr], 3);
    double dmax = 2*range_sigma;
    double dmin = dmax;
    int x, y;
    int r = xyz_score_window;
    for (x = xi[i_arr] - r; x<=xi[i_arr] + r; ++x) {
      for (y = yi[i_arr] - r; y <= yi[i_arr] + r; ++y) {
	if (x >= 0 && x < (range_image_data.w) && y>=0 && y<(range_image_data.h) && range_image_cnt.ptr[x * range_image_cnt.m + y] > 0) {
	  double obs_range = range_image.ptr[x * range_image.m + y];
	  double d = fabs(model_range - obs_range);
	  if (d < dmin) 
	    dmin = d;	    
	}
      }
    }
    double d = dmin;
    xyz_score[i_arr] = vis_pmf[i_arr] * log(cu_normpdf(d, 0, range_sigma));
    
  }
}

__global__ void cu_compute_xyz_score_final(double *xyz_scores, int num_samples, double *b_xyz, scope_params_t *params, int score_round) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  xyz_scores[i] -= log(cu_normpdf(0, 0, params->range_sigma));

  if ((score_round == 2 && params->score2_use_score_comp_models) || (score_round == 3 && params->score3_use_score_comp_models))
    xyz_scores[i] = cu_logistic(xyz_scores[i], b_xyz);

  double w = 0;
  if (score_round == 2)
    w = params->score2_xyz_weight;
  else
    w = params->score3_xyz_weight;

  xyz_scores[i] *= w;
}

__global__ void cu_compute_normal_score_individual(double *normal_score, double *wtot_individual, double *cloud_normals, double *vis_pmf, scope_noise_model_t *noise_models, int num_samples, 
						int num_validation_points, int *xi, int *yi, cu_int_matrix_t range_image_cnt, cu_double_matrix3d_t range_image_normals, scope_params_t *params, int score_round)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= num_validation_points || i >= num_samples)
    return;

  int i_arr = j + i * num_validation_points;

  //TODO: make this a param
  double normalvar_thresh = params->normalvar_thresh;

  normal_score[i_arr] = 0.0;
  wtot_individual[i_arr] = 0.0;

  if (vis_pmf[i_arr] > .01/ (double) num_validation_points && noise_models[i_arr].normal_sigma <= normalvar_thresh) {
    double normal_sigma = params->normal_sigma * noise_models[i_arr].normal_sigma;
    double dmax = 2*normal_sigma;
    double d = dmax;
    if ((xi[i_arr] != -1 && yi[i_arr] != -1) && range_image_cnt.ptr[xi[i_arr] * range_image_cnt.m + yi[i_arr]] > 0) {
      d = 1.0 - cu_dot(&cloud_normals[3*i_arr], &(range_image_normals.ptr[xi[i_arr] * range_image_normals.m * range_image_normals.p + yi[i_arr] * range_image_normals.p]), 3);
      d = MIN(d, dmax);
    }
    normal_score[i_arr] = vis_pmf[i_arr] * log(cu_normpdf(d, 0, normal_sigma));
    wtot_individual[i_arr] = vis_pmf[i_arr];
  }
}

__global__ void cu_compute_normal_score_final(double *normal_scores, double *wtot, int num_samples, double *b_normal, scope_params_t *params, int score_round) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;
  
  if (wtot[i] > 0.0)
    normal_scores[i] /= wtot[i];
  normal_scores[i] -= log(cu_normpdf(0, 0, params->normal_sigma));

  if ((score_round == 2 && params->score2_use_score_comp_models) || (score_round == 3 && params->score3_use_score_comp_models))
    normal_scores[i] = cu_logistic(normal_scores[i], b_normal);

  double w = 0;
  if (score_round == 2)
    w = params->score2_normal_weight;
  else
    w = params->score3_normal_weight;

  normal_scores[i] *= w;
}

__global__ void cu_compute_vis_score(double *vis_score, double *vis_sums, int n, scope_params_t *params, int score_round)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= n)
    return;

  vis_score[i] = log(vis_sums[i] / (double) n);
  
  double w = 0;
  if (score_round == 2)
    w = params->score2_vis_weight;
  else
    w = params->score3_vis_weight;

  vis_score[i] *= w;
}

__global__ void cu_set_mask_for_segment_affinity(int *mask, int *segments, int *num_segments, int num_obs_segments, int num_samples) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= num_segments[i])
    return;

  // Assumes mask is initialized to all zeros before kernel execution  
  mask[segments[j + i * num_obs_segments] + i * num_obs_segments] = 1;
}

// compute the segment affinity score for a scope sample
__global__ void cu_compute_segment_affinity_score_per_seg(double *seg_affinity_score_per_seg, int *segments, int *num_segments, cu_double_matrix_t segment_affinities, int num_obs_segments, int *mask, 
							  int num_samples)	    
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= num_obs_segments)
    return;

  int k;
  
  seg_affinity_score_per_seg[j + i * num_obs_segments] = 0.0;
  if (mask[j + i * num_obs_segments] == 0) {
    for (k = 0; k < num_segments[i]; ++k) {
      int s = segments[k + i * num_obs_segments];
      double a = MIN(segment_affinities.ptr[s * segment_affinities.m + j], .9);
      if (a > 0.5)
	seg_affinity_score_per_seg[j + i * num_obs_segments] += log((1-a)/a);
    }
  }
}

__global__ void cu_compute_segment_affinity_score_final(double *seg_affinity_score, scope_params_t *params, int score_round, int num_samples) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_samples)
    return;
    
  seg_affinity_score[i] *= .05;

  double weight = 0;
  if (score_round == 2)
    weight = params->score2_segment_affinity_weight;
  else
    weight = params->score3_segment_affinity_weight;

  seg_affinity_score[i] *= weight;
}

__global__ void cu_generate_n_for_range_edge(int *n_out, int *vi, int num_samples, int num_validation_points, cu_int_arr_t range_edges_view_cnt) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;  
  
  if (i >= num_samples)
    return;
  
  int v_idx = vi[i];
  int num_edge_points = range_edges_view_cnt.ptr[v_idx];

  int n = num_validation_points;

  if (n >= num_edge_points || n == 0) {
    n = num_edge_points;
  }
  n_out[i] = n;
}

__global__ void cu_get_range_edge_idx(int *idx, int *needed, int num_samples, int total_pts, int n, uint *rands, int *vi,cu_int_arr_t range_edges_view_idx)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= needed[i])
    return;

  // NOTE(sanja): This might need some fixing if I use the function in a broader sense, like on a CPU version
  if (needed[i] <= n) {  // use all the points
    idx[j + i * total_pts] = j;
  } else {
    idx[j + i * total_pts] = ((rands[i << 1] % needed[i]) + (j * (big_primes[rands[(i << 1) + 1] % 100] % needed[i]))) % needed[i];
  }
  int vp_idx = range_edges_view_idx.ptr[vi[i]];
  idx[j + i * total_pts] += vp_idx;

}

/*__global__ void cu_get_range_edge_points(double *P, int num_samples, int *n, int *idx, int n_edge, cu_double_matrix_t range_edges_points)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples)
    return;
  if (j >= n[i])
    return;

  // get the actual points in the correct pose
  P[3 * i * n_edge + 3 * j] = range_edges_points.ptr[3 * idx[j + i * n_edge]];
  P[3 * i * n_edge + 3 * j + 1] = range_edges_points.ptr[3 * idx[j + i * n_edge] + 1];
  P[3 * i * n_edge + 3 * j + 2] = range_edges_points.ptr[3 * idx[j + i * n_edge] + 2];
  }*/

__global__ void cu_compute_edge_score_individual(double *edge_score, double *vis_pmf, int *xi, int *yi, cu_double_matrix_t edge_image, int num_samples, int *n, int n_edge) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= n[i])
    return;

  edge_score[j + i * n_edge] = 0.0;
  if (xi[j + i *n_edge] != -1 && yi[j + i * n_edge] != -1) {
    edge_score[j + i * n_edge] = vis_pmf[j + i * n_edge] * edge_image.ptr[xi[j + i *n_edge]*edge_image.m + yi[j + i *n_edge]];
  }
}

__global__ void cu_compute_edge_score_final(double *edge_score, double *vis_score, double *vis_prob_sums, double *occ_score, int num_samples, int *n_arr, double *b_edge, double *b_edge_occ, 
					    scope_params_t *params, int score_round) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_samples)
    return;

  if ((score_round == 2 && params->score2_use_score_comp_models) || (score_round == 3 && params->score3_use_score_comp_models)) {
    edge_score[i] = cu_logistic(edge_score[i], b_edge);
    if (occ_score)
      occ_score[i] = cu_logistic(occ_score[i], b_edge_occ);
  }

  vis_score[i] = log(vis_prob_sums[i] / (double) n_arr[i]);

  double w1=0.0, w2=0.0, w3=0.0;
  w1=1.0, w2=1.0, w3=1.0;
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

  if (occ_score)
    edge_score[i] = (w1 * edge_score[i]) + (w2 * vis_score[i]) + (w3 * occ_score[i]);
  else
    edge_score[i] = (w1 * edge_score[i]) + (w2 * vis_score[i]);    
}
  
__global__ void cu_score_round1(double *scores, int *xi, int *yi, double *cloud, cu_double_matrix_t range_image, int num_samples, int num_validation_points) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  double dthresh = .05;  //TODO: make this a param
  double sample_score = 0;

  double dest[3];
  
  int i_arr = i*num_validation_points;
  int j;

  // TODO(sanja): optimize!
  for (j = 0; j < num_validation_points; ++j) {
    dest[0] = cloud[3*(i_arr + j)]; dest[1] = cloud[3*(i_arr + j)+1]; dest[2] = cloud[3*(i_arr + j) + 2];
    if ((xi[i_arr + j] != -1 && yi[i_arr + j] != -1) && 
	range_image.ptr[xi[i_arr + j]*range_image.m + yi[i_arr + j]] > dthresh + cu_norm(dest, 3))
      sample_score -= 1.0;
  }

  sample_score /= (double)num_validation_points;
  
  scores[i] = sample_score;
}
 

// TODO(sanja): make this a more general function that takes double** or something like that
__global__ void cu_add_all_scores(double *cu_scores, double *cu_xyz_score, double *cu_normal_score, double *cu_vis_score, double *cu_seg_affinity_score, double *cu_edge_scores, int num_samples) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  cu_scores[i] = cu_xyz_score[i] + cu_normal_score[i] + cu_vis_score[i] + cu_seg_affinity_score[i] + cu_edge_scores[i];
}

__global__ void cu_add_3_scores(double *cu_scores, double *cu_xyz_score, double *cu_normal_score, double *cu_edge_scores, int num_samples) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  cu_scores[i] = cu_xyz_score[i] + cu_normal_score[i] + cu_edge_scores[i];
}

void get_vis_prob_sums_and_pmf(double *vis_prob, double *vis_prob_sums, double *vis_pmf, double *cloud, double *normals, int *xi, int *yi, cu_double_matrix_t range_image, cu_range_image_data_t range_image_data, 
			       int vis_pixel_radius, int num_samples, int n, int *n_arr, scope_params_t *params, dim3 block, dim3 thread, dim3 block_sum, dim3 thread_sum, int slow_sum) {
  cu_compute_visibility_prob<<<block, thread>>>(vis_prob, cloud, normals, xi, yi, range_image_data, range_image, params->vis_thresh, vis_pixel_radius, num_samples, n, n_arr);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "vis_prob!\n" );
  if (slow_sum)
    cu_add_matrix_rows_slow<<<block_sum, thread_sum>>>(vis_prob_sums, vis_prob, num_samples, n, n_arr);
  else
    cu_add_matrix_rows_medium<<<block_sum, thread_sum, thread_sum.x * sizeof(double)>>>(vis_prob_sums, vis_prob, num_samples, n, n_arr); 
  // TODO(sanja): Optimize. ArrayFire?
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Vis prob sums!\n" );
  cu_divide_matrix_with_vector<<<block, thread>>>(vis_pmf, vis_prob, vis_prob_sums, num_samples, n, n_arr);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Vis pmf!\n" );    
}

void unpack_x_q(double *cu_x, double *cu_q, scope_sample_t *samples, int num_samples) {
  double **samples_x = new_matrix2(num_samples, 3);  
  double **samples_q = new_matrix2(num_samples, 4);
  int i;
  for (i = 0; i < num_samples; ++i) {
    memcpy(samples_x[i], samples[i].x, 3 * sizeof(double));
  }
  for (i = 0; i < num_samples; ++i) {
    memcpy(samples_q[i], samples[i].q, 4 * sizeof(double));
  }  
  
  cudaMemcpy(cu_x, samples_x[0], 3 * num_samples * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cu_q, samples_q[0], 4 * num_samples * sizeof(double), cudaMemcpyHostToDevice);
  
  free_matrix2(samples_x);
  free_matrix2(samples_q);
}

void get_range_edge_points(int *cu_n, double *cu_P, int num_samples, int n_edge, int *n_arr, int num_validation_points, cu_model_data_t *cu_model, int *cu_vi, int find_vi, double *cu_x, double *cu_q,
			   dim3 block, dim3 thread, dim3 block_small, dim3 thread_small) {
  if (find_vi) {
    cu_get_viewpoints<<<block_small, thread_small>>>(cu_vi, num_samples, cu_x, cu_q, cu_model->range_edges_model_views);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Viewpoints!\n" );
  }	
  
  int *cu_idx_edge;
  cu_malloc(&cu_idx_edge, num_samples * n_edge * sizeof(int), "idx_edge");
  uint *cu_rands_edge;
  cu_malloc(&cu_rands_edge, 2 * num_samples * sizeof(uint), "rands");
  cu_generate_n_for_range_edge<<<block_small, thread_small>>>(cu_n, cu_vi, num_samples, num_validation_points, cu_model->range_edges_view_cnt);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "n!\n" );
  curandGenerate(gen, cu_rands_edge, 2*num_samples);
  cu_get_range_edge_idx<<<block, thread>>>(cu_idx_edge, cu_n, num_samples, n_edge, num_validation_points, cu_rands_edge, cu_vi, cu_model->range_edges_view_idx);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "idx edge!\n" );
  
  cu_reorder_rows<<<block, thread>>>(cu_P, cu_model->range_edges_points, cu_idx_edge, num_samples, n_edge, 3, cu_n);
  //cu_get_range_edge_points<<<block, thread>>>(cu_P, num_samples, cu_n, cu_idx_edge, n_edge, cu_model->range_edges_points);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "edge pts\n" );
  cu_free(cu_rands_edge, "rands_edge");
  cu_free(cu_idx_edge, "idx_edge");
}

void get_validation_points(int *cu_idx, int model_points, int num_validation_points, int num_samples, dim3 block, dim3 thread) {
  uint *cu_rands;
  cu_malloc(&cu_rands, 2 * num_samples * sizeof(uint), "rands");
  if (model_points > num_validation_points) {
    curandGenerate(gen, cu_rands, 2*num_samples);
  }
  cu_get_validation_points<<<block, thread>>>(cu_idx, model_points, num_validation_points, num_samples, cu_rands);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Validation!\n" );
  cu_free(cu_rands, "rands free");
}

void compute_xyz_score(double *cu_xyz_score, double *cu_cloud, int *cu_xi, int *cu_yi, double *cu_vis_pmf, scope_noise_model_t *cu_noise_models, cu_double_matrix_t range_image,
		       cu_range_image_data_t range_image_data, cu_int_matrix_t range_image_cnt, int num_samples, int num_validation_points, scope_params_t *cu_params, int round,
		       double *b_xyz, dim3 block_size, dim3 threads_per_block, dim3 block_size_sum, dim3 thread_size_sum, dim3 block_size_small, dim3 thread_size_small) {
  int num_total = num_samples * num_validation_points;

  double *cu_xyz_score_per_point;
  cu_malloc(&cu_xyz_score_per_point, num_total * sizeof(double), "xyz_scores_pp");
  cu_compute_xyz_score_individual<<<block_size, threads_per_block>>>(cu_xyz_score_per_point, cu_cloud, cu_xi, cu_yi, cu_vis_pmf, cu_noise_models, num_samples, num_validation_points, 
								     range_image, range_image_data, range_image_cnt, cu_params);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "xyz individual!\n" );
  cu_add_matrix_rows_medium<<<block_size_sum, thread_size_sum, thread_size_sum.x * sizeof(double)>>>(cu_xyz_score, cu_xyz_score_per_point, num_samples, num_validation_points, NULL);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "xyz sums!\n" );
  
  cu_compute_xyz_score_final<<<block_size_small, thread_size_small>>>(cu_xyz_score, num_samples, b_xyz, cu_params, round);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "xyz final!\n" );
  cu_free(cu_xyz_score_per_point, "xyz_scores_pp");
}

void compute_normal_score(double *cu_normal_score, double *cu_normals, double *cu_vis_pmf, scope_noise_model_t *cu_noise_models, int num_samples, int num_validation_points, int *cu_xi, int *cu_yi,
			  cu_int_matrix_t range_image_cnt, cu_double_matrix3d_t range_image_normals, double *b_normal, scope_params_t *cu_params, int round,
			  dim3 block_size, dim3 threads_per_block, dim3 block_size_sum, dim3 thread_size_sum, dim3 block_size_small, dim3 thread_size_small) {
  int num_total = num_samples * num_validation_points;

  double *cu_normal_score_per_point;
  cu_malloc(&cu_normal_score_per_point, num_total * sizeof(double), "normal_score_pp");
  double *cu_wtot_per_point;
  cu_malloc(&cu_wtot_per_point, num_total * sizeof(double), "wtot_pp");
  cu_compute_normal_score_individual<<<block_size, threads_per_block>>>(cu_normal_score_per_point, cu_wtot_per_point, cu_normals, cu_vis_pmf, cu_noise_models, num_samples, num_validation_points, cu_xi, cu_yi,
									range_image_cnt, range_image_normals, cu_params, round);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "normal individual!\n" );
  double *cu_wtot;
  cu_malloc(&cu_wtot, num_samples * sizeof(double), "wtot");
  cu_add_matrix_rows_medium<<<block_size_sum, thread_size_sum, thread_size_sum.x * sizeof(double)>>>(cu_normal_score, cu_normal_score_per_point, num_samples, num_validation_points, NULL);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "add 1!\n" );
  cu_add_matrix_rows_medium<<<block_size_sum, thread_size_sum, thread_size_sum.x * sizeof(double)>>>(cu_wtot, cu_wtot_per_point, num_samples, num_validation_points, NULL);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "add 2!\n" );
  
  cu_compute_normal_score_final<<<block_size_small, thread_size_small>>>(cu_normal_score, cu_wtot, num_samples, b_normal, cu_params, round);
  cu_free(cu_normal_score_per_point, "normal_scores_pp");
  cu_free(cu_wtot_per_point, "wtot_pp");
  cu_free(cu_wtot, "wtot"); 
}

void compute_edge_score(double *cu_edge_scores, double *cu_P, cu_range_image_data_t range_image_data, cu_double_matrix_t range_image, int num_samples, int n_edge, int *cu_n, cu_double_matrix_t edge_image, 
			double *b_edge, double *b_edge_occ, scope_params_t *params, scope_params_t *cu_params, int round, dim3 block_size_n_edge, dim3 block_size_sum, dim3 thread_size_sum, 
			dim3 block_size_small, dim3 thread_size_small)
{

  double *cu_vis_prob_edge, *cu_vis_prob_sums_edge, *cu_vis_pmf_edge;
  cu_malloc(&cu_vis_prob_edge, num_samples * n_edge * sizeof(double), "vis_prob_edge");
  cu_malloc(&cu_vis_prob_sums_edge, num_samples * sizeof(double), "vis_prob_sums_edge");
  cu_malloc(&cu_vis_pmf_edge, num_samples * n_edge * sizeof(double), "vis_pmf_edge");
  int *cu_xi_edge;
  cu_malloc(&cu_xi_edge, num_samples * n_edge * sizeof(int), "xi");
  int *cu_yi_edge;
  cu_malloc(&cu_yi_edge, num_samples * n_edge * sizeof(int), "yi");
  cu_populate_xi_yi<<<block_size_n_edge, thread_size_sum>>>(cu_xi_edge, cu_yi_edge, cu_P, range_image_data, num_samples, n_edge, cu_n);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "edge xi yi!\n" );
  
  int vis_pixel_radius = 2;
  
  get_vis_prob_sums_and_pmf(cu_vis_prob_edge, cu_vis_prob_sums_edge, cu_vis_pmf_edge, cu_P, NULL, cu_xi_edge, cu_yi_edge, range_image, range_image_data, vis_pixel_radius, num_samples, n_edge,
			    cu_n, params, block_size_n_edge, thread_size_sum, block_size_small, thread_size_small, 1);
  
  double *cu_edge_score_individual;
  cu_malloc(&cu_edge_score_individual, num_samples * n_edge * sizeof(double), "edge_score");
  cu_compute_edge_score_individual<<<block_size_sum, thread_size_sum>>>(cu_edge_score_individual, cu_vis_pmf_edge, cu_xi_edge, cu_yi_edge, edge_image, num_samples, cu_n, n_edge);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "edge score individual!\n" );
  cu_add_matrix_rows_slow<<<block_size_small, thread_size_small>>>(cu_edge_scores, cu_edge_score_individual, num_samples, n_edge, cu_n);
  
  double *cu_vis_scores;
  cu_malloc(&cu_vis_scores, num_samples * sizeof(double), "vis_scores");
  cu_compute_edge_score_final<<<block_size_small, thread_size_small>>>(cu_edge_scores, cu_vis_scores, cu_vis_prob_sums_edge, NULL, num_samples, cu_n, b_edge, b_edge_occ, cu_params, round);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "edge score final!\n" );
  
  cu_free(cu_edge_score_individual, "edge_score_individual");
  cu_free(cu_vis_scores, "vis_scores");
  cu_free(cu_vis_prob_sums_edge, "vis_prob_sums_edge");
  cu_free(cu_vis_pmf_edge, "vis_pmf_edge");
  cu_free(cu_vis_prob_edge, "vis_prob_edge");
  cu_free(cu_xi_edge, "xi_edge");
  cu_free(cu_yi_edge, "yi_edge");

}

void score_samples(double *scores, scope_sample_t *samples, int num_samples, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, scope_params_t *cu_params, scope_params_t *params, int num_validation_points, 
		   int model_points, int num_obs_segments, int edge_scoring, int round) {
  // NEXT(sanja): Figure out why the hell seg mask makes things crappy

  dim3 threads_per_block(256, 1, 1);
  dim3 block_size(ceil(1.0 * num_validation_points / threads_per_block.x), num_samples);

  dim3 thread_size_small(64);
  dim3 block_size_small(ceil(1.0 * num_samples/thread_size_small.x));

  dim3 thread_size_sum(256);
  dim3 block_size_sum(1, num_samples);
  dim3 thread_size_sum_small(64);
  
  int num_total = num_samples * num_validation_points;

  double *cu_samples_x;
  cu_malloc(&cu_samples_x, num_samples * 3 * sizeof(double), "samples_x");
  double *cu_samples_q;
  cu_malloc(&cu_samples_q, num_samples * 4 * sizeof(double), "samples_y");
  unpack_x_q(cu_samples_x, cu_samples_q, samples, num_samples);

  int i;

  int *cu_idx;
  cu_malloc(&cu_idx, num_total * sizeof(int), "idxs");
  get_validation_points(cu_idx, model_points, num_validation_points, num_samples, block_size, threads_per_block);

  // extract transformed model validation features
  double *cu_cloud;
  cu_malloc(&cu_cloud, 3 * num_total * sizeof(double), "cloud");
  cu_get_sub_cloud_at_pose<<<block_size, threads_per_block>>>(cu_cloud, cu_model->points, cu_samples_x, cu_samples_q, cu_idx, num_samples, num_validation_points);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Subcloud!\n" );

  int *cu_xi;
  cu_malloc(&cu_xi, num_total * sizeof(int), "xi");
  int *cu_yi;
  cu_malloc(&cu_yi, num_total * sizeof(int), "yi");
  cu_populate_xi_yi<<<block_size, threads_per_block>>>(cu_xi, cu_yi, cu_cloud, cu_obs->range_image_data, num_samples, num_validation_points, NULL);

  double *cu_scores;
  cu_malloc(&cu_scores, num_samples * sizeof(double), "scores");

  if (round == 1) {
    cu_score_round1<<<block_size_small, thread_size_small>>>(cu_scores, cu_xi, cu_yi, cu_cloud, cu_obs->range_image, num_samples, num_validation_points);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Round 1 score!\n" );
  } else {    
    double *cu_normals;
    cu_malloc(&cu_normals, 3 * num_total * sizeof(double), "normals");
    cu_get_sub_cloud_normals_rotated<<<block_size, threads_per_block>>>(cu_normals, cu_model->normals, cu_samples_q, cu_idx, num_samples, num_validation_points);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Normals!\n" );

    double *cu_vis_prob;
    cu_malloc(&cu_vis_prob, num_total * sizeof(double), "vis_prob");
    double *cu_vis_prob_sums;
    cu_malloc(&cu_vis_prob_sums, num_samples * sizeof(double), "vis_prob_sums");
    double *cu_vis_pmf;
    cu_malloc(&cu_vis_pmf, num_total * sizeof(double), "vis_pmf");
    get_vis_prob_sums_and_pmf(cu_vis_prob, cu_vis_prob_sums, cu_vis_pmf, cu_cloud, cu_normals, cu_xi, cu_yi, cu_obs->range_image, cu_obs->range_image_data, 0, num_samples, num_validation_points, NULL, params,
			      block_size, threads_per_block, block_size_sum, thread_size_sum, 0);

    int *cu_vi;
    cu_malloc(&cu_vi, num_samples * sizeof(int), "vi");
    cu_get_viewpoints<<<block_size_small, thread_size_small>>>(cu_vi, num_samples, cu_samples_x, cu_samples_q, cu_model->range_edges_model_views);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Viewpoints!\n" );

    scope_noise_model_t *cu_noise_models;
    cu_malloc(&cu_noise_models, num_total * sizeof(scope_noise_model_t), "noise_models");
    cu_get_noise_models<<<block_size, threads_per_block>>>(cu_noise_models, cu_cloud, cu_normals, cu_idx, cu_vi, cu_model->ved, cu_model->normalvar, num_samples, 
							   num_validation_points);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Noise model!\n" );    
    // TODO(sanja): Save results before weights kick in
    double *cu_xyz_score;
    cu_malloc(&cu_xyz_score, num_samples * sizeof(double), "xyz_scores");

    compute_xyz_score(cu_xyz_score, cu_cloud, cu_xi, cu_yi, cu_vis_pmf, cu_noise_models, cu_obs->range_image, cu_obs->range_image_data, cu_obs->range_image_cnt, num_samples, num_validation_points,
		      cu_params, round,	cu_model->score_comp_models->b_xyz, block_size, threads_per_block, block_size_sum, thread_size_sum, block_size_small, thread_size_small);
    
    double *cu_normal_score;
    cu_malloc(&cu_normal_score, num_samples * sizeof(double), "normal_score");

    compute_normal_score(cu_normal_score, cu_normals, cu_vis_pmf, cu_noise_models, num_samples, num_validation_points, cu_xi, cu_yi, cu_obs->range_image_cnt, cu_obs->range_image_normals, 
			 cu_model->score_comp_models->b_normal, cu_params, round, block_size, threads_per_block, block_size_sum, thread_size_sum, block_size_small, thread_size_small);

    double *cu_vis_score;
    cu_malloc(&cu_vis_score, num_samples * sizeof(double), "vis_score");
    cu_compute_vis_score<<<block_size_small, thread_size_small>>>(cu_vis_score, cu_vis_prob_sums, num_validation_points, cu_params, round);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "vis score!\n" );

    // TODO(sanja): Figure out how to speed up the prep for segment calculation
    double *cu_seg_affinity_score_per_seg;
    cu_malloc(&cu_seg_affinity_score_per_seg, num_samples * num_obs_segments * sizeof(double), "seg_aff_per_seg");
    int *cu_mask;
    cu_malloc(&cu_mask, num_samples * num_obs_segments * sizeof(int), "mask");
    cudaMemset(cu_mask, 0, num_samples * num_obs_segments * sizeof(int));
    int *num_segments;
    safe_calloc(num_segments, num_samples, int);
    for (i = 0; i < num_samples; ++i) {
      num_segments[i] = samples[i].num_segments;
    }
    int *cu_num_segments;
    cu_malloc(&cu_num_segments, num_samples * sizeof(int), "num_segments");
    cudaMemcpy(cu_num_segments, num_segments, num_samples * sizeof(int), cudaMemcpyHostToDevice);
    free(num_segments);
    int *tmp_segments_idx;
    safe_malloc(tmp_segments_idx, num_samples * num_obs_segments, int);
    memset(tmp_segments_idx, -1, num_samples * num_obs_segments * sizeof(int));
    for (i = 0; i < num_samples; ++i) {
      memcpy(&(tmp_segments_idx[i * num_obs_segments]), samples[i].segments_idx, samples[i].num_segments * sizeof(int));
    }
    int *cu_segments_idx;
    cu_malloc(&cu_segments_idx, num_samples * num_obs_segments * sizeof(int), "segments_idx");
    cudaMemcpy(cu_segments_idx, tmp_segments_idx, num_samples * num_obs_segments * sizeof(int), cudaMemcpyHostToDevice);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "seg idx memcpy!\n" );

    free(tmp_segments_idx);
    double *cu_seg_affinity_score;
    cu_malloc(&cu_seg_affinity_score, num_samples * sizeof(double), "seg_aff_per_seg");
    
    dim3 block_size_seg(ceil(1.0 * num_obs_segments / thread_size_sum.x), num_samples);
    //cu_set_mask_for_segment_affinity<<<block_size_seg, thread_size_sum>>>(cu_mask, cu_segments_idx, cu_num_segments, num_obs_segments, num_samples);
    cu_set_mask_for_segment_affinity<<<block_size_seg, thread_size_sum>>>(cu_mask, cu_segments_idx, cu_num_segments, num_obs_segments, num_samples);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "seg mask!\n" );

    cu_compute_segment_affinity_score_per_seg<<<block_size, thread_size_small>>>(cu_seg_affinity_score_per_seg, cu_segments_idx, cu_num_segments, cu_obs->segment_affinities, num_obs_segments, 
										 cu_mask, num_samples);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "seg per seg!\n" );

    cu_add_matrix_rows_slow<<<block_size_small, thread_size_small>>>(cu_seg_affinity_score, cu_seg_affinity_score_per_seg, num_samples, num_obs_segments, NULL);
    cu_compute_segment_affinity_score_final<<<block_size_small, thread_size_small>>>(cu_seg_affinity_score, cu_params, round, num_samples);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "seg affinity!\n" );

    double *cu_edge_scores;
    cu_malloc(&cu_edge_scores, num_samples * sizeof(double), "edge_scores");
    cudaMemset(cu_edge_scores, 0, num_samples * sizeof(double));
    if ( cudaSuccess != cudaGetLastError() )
      printf( "memset!\n" );
    if (edge_scoring) {

      int n_edge = cu_model->max_num_edges;
      int *cu_n;
      cu_malloc(&cu_n, num_samples * sizeof(int), "n");
      dim3 block_size_n_edge(ceil(1.0 * n_edge / thread_size_sum.x), num_samples);
      double *cu_P;
      cu_malloc(&cu_P, num_samples * n_edge * 3*sizeof(double), "cu_P");
      get_range_edge_points(cu_n, cu_P, num_samples, n_edge, cu_n, num_validation_points, cu_model, cu_vi, 0, NULL, NULL, block_size_n_edge, thread_size_sum, block_size_small, thread_size_small);

      cu_transform_cloud<<<block_size_n_edge, thread_size_sum>>>(cu_P, cu_P, cu_samples_x, cu_samples_q, num_samples, n_edge, cu_n);
      if ( cudaSuccess != cudaGetLastError() )
	printf( "transform cloud\n" );

      compute_edge_score(cu_edge_scores, cu_P, cu_obs->range_image_data, cu_obs->range_image, num_samples, n_edge, cu_n, cu_obs->edge_image, cu_model->score_comp_models->b_edge, 
			 cu_model->score_comp_models->b_edge_occ, params, cu_params, round, block_size_n_edge, block_size_sum, thread_size_sum, block_size_small, thread_size_small);
      
      cu_free(cu_n, "n");
      cu_free(cu_P, "P");
    }
    
    cu_add_all_scores<<<block_size_small, thread_size_small>>>(cu_scores, cu_xyz_score, cu_normal_score, cu_vis_score, cu_seg_affinity_score, cu_edge_scores, num_samples);
      
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Final addition!\n" );

    // NEXT(sanja): Make calls for each score component async.

    cu_free(cu_normals, "normals");
    cu_free(cu_vis_prob, "vis_prob");
    cu_free(cu_vis_prob_sums, "vis_prob_sums");
    cu_free(cu_vis_pmf, "vis_pmf");
    cu_free(cu_vi, "vi");
    cu_free(cu_noise_models, "noise_models");

    cu_free(cu_xyz_score, "xyz_scores");
    cu_free(cu_normal_score, "normal_scores");
    cu_free(cu_vis_score, "vis_score");
    cu_free(cu_seg_affinity_score_per_seg, "seg_aff_per_seg");
    cu_free(cu_seg_affinity_score, "seg_aff");
    cu_free(cu_mask, "mask");
    cu_free(cu_num_segments, "num_segments");
    cu_free(cu_segments_idx, "segments_idx");
    cu_free(cu_edge_scores, "edge_scores");
  }

  cudaMemcpy(scores, cu_scores, num_samples * sizeof(double), cudaMemcpyDeviceToHost);
  
  cu_free(cu_samples_x, "samples_x"); cu_free(cu_samples_q, "samples_y");
  cu_free(cu_idx, "idx");
  cu_free(cu_cloud, "cloud");
  cu_free(cu_xi, "xi"); cu_free(cu_yi, "yi");
  cu_free(cu_scores, "scores");
  cudaDeviceSynchronize();
}

__device__ void cu_matrix_cell_gradient(double *g, int i, int j, double *X, int n, int m)
{
  
  if (i == 0)
    g[0] = X[1*m + j] - X[0*m + j];
  else if (i == n-1)
    g[0] = X[(n-1)*m + j] - X[(n-2)*m + j];
  else
    g[0] = (X[(i+1)*m + j] - X[(i-1) * m + j]) / 2.0;

  if (j == 0)
    g[1] = X[i*m + 1] - X[i*m + 0];
  else if (j == m-1)
    g[1] = X[i*m + m-1] - X[i*m + m-2];
  else
    g[1] = (X[i*m + j+1] - X[i*m + j-1]) / 2.0;
}

// get the jacobian of R*x w.r.t. q
__device__ void cu_point_rotation_jacobian(double out[][4], double *q, double *x)
{
  double q1 = q[0];
  double q2 = q[1];
  double q3 = q[2];
  double q4 = q[3];
  
  double v1 = x[0];
  double v2 = x[1];
  double v3 = x[2];
  
  out[0][0] = 2*(q1*v1 + q3*v3 - q4*v2); out[0][1] = 2*(q2*v1 + q3*v2 + q4*v3); out[0][2] = 2*(q1*v3 + q2*v2 - q3*v1); out[0][3] = 2*(q2*v3 - q1*v2 - q4*v1);
  out[1][0] = 2*(q1*v2 - q2*v3 + q4*v1); out[1][1] = 2*(q3*v1 - q2*v2 - q1*v3); out[1][2] = 2*(q2*v1 + q3*v2 + q4*v3); out[1][3] = 2*(q1*v1 + q3*v3 - q4*v2);
  out[2][0] = 2*(q1*v3 + q2*v2 - q3*v1); out[2][1] = 2*(q1*v2 - q2*v3 + q4*v1); out[2][2] = 2*(q4*v2 - q3*v3 - q1*v1); out[2][3] = 2*(q2*v1 + q3*v2 + q4*v3);
}

__device__ void cu_range_image_pixel_pose_jacobian(double dij_dxq[][7], cu_range_image_data_t range_image_data, double *model_point, double *x, double *q)
{
  // transform model point by model pose (x,q)
  double p[3];
  double R[3][3];
  cu_quaternion_to_rotation_matrix(R,q);
  cu_matrix_vec_mult_3(p, R, model_point, 3);
  cu_add(p, p, x, 3);
  double p1 = p[0];
  double p2 = p[1];
  double p3 = p[2];
  double r = cu_norm(p,3);

  // compute jacobian of (i,j) w.r.t. p (and x)
  double ci = 1.0 / (p1*p1 + p3*p3);
  double cj = r * sqrt(ci);
  double dij_dp[2][3] = {{ci*p3, 0, -ci*p1},  {cj*p1*p2, cj*(p2*p2-r*r), cj*p2*p3}};

  // compute jacobian of (i,j) w.r.t. q
  double dp_dq[3][4];
  cu_point_rotation_jacobian(dp_dq, q, model_point);
  double dij_dq[2][4];
  cu_matrix_mult_2_3_4(dij_dq, dij_dp, dp_dq);

  // copy jacobians into dij_dxq
  dij_dxq[0][0] = dij_dp[0][0]; dij_dxq[0][1] = dij_dp[0][1]; dij_dxq[0][2] = dij_dp[0][2];
  dij_dxq[1][0] = dij_dp[1][0]; dij_dxq[1][1] = dij_dp[1][1]; dij_dxq[1][2] = dij_dp[1][2];

  dij_dxq[0][3] = dij_dq[0][0]; dij_dxq[0][4] = dij_dq[0][1]; dij_dxq[0][5] = dij_dq[0][2]; dij_dxq[0][6] = dij_dq[0][3];
  dij_dxq[1][3] = dij_dq[1][0]; dij_dxq[1][4] = dij_dq[1][1]; dij_dxq[1][5] = dij_dq[1][2]; dij_dxq[1][6] = dij_dq[1][3];

  // divide by range image resolution
  cu_mult(dij_dxq[0], dij_dxq[0], 1.0/range_image_data.res, 7);
  cu_mult(dij_dxq[1], dij_dxq[1], 1.0/range_image_data.res, 7); // NOTE(sanja): This can probably be one multiplication of length 14, but I am somewhat defensive right now
}

__global__ void cu_edge_score_gradient_individual(double *edge_gradient_score_individual, double *dI_dxq_individual, double *vis_prob, double *vis_pmf, int *xi, int *yi, double *P, double *x, double *q,
					       cu_double_matrix_t edge_image, cu_range_image_data_t range_image_data, cu_double_matrix_t range_image, int num_samples, int n, int *n_arr) {

  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= n_arr[i])
    return;

  int i_arr = j + i * n;

  edge_gradient_score_individual[i_arr] = 0.0;
  int k;
  for (k = 0; k < 7; ++k)
    dI_dxq_individual[k + 7*i_arr] = 0.0;

  if (vis_prob[i_arr] < .01)
    return;

  if (xi[i_arr] == -1 || yi[i_arr] == -1)
    return;
  
  // add pixel edge score to total score
  edge_gradient_score_individual[i_arr] = vis_pmf[i_arr] * edge_image.ptr[xi[i_arr] * edge_image.m + yi[i_arr]];
  
  // get edge image gradient at current pixel
  double dI_dij[2];
  cu_matrix_cell_gradient(dI_dij, xi[i_arr], yi[i_arr], edge_image.ptr, range_image_data.w, range_image_data.h);
    
  // get gradient of pixel location w.r.t. model pose (x,q)
  double dij_dxq[2][7];
  cu_range_image_pixel_pose_jacobian(dij_dxq, range_image_data, &P[3*i_arr], &x[3 * i], &q[4*i]);
  
  // get gradient of this point's edge score w.r.t. model pose (x,q)
  double dI_dxq[7];
  cu_vec_matrix_mult_7(dI_dxq, dI_dij, dij_dxq, 2);
  
  cu_mult(&dI_dxq_individual[7*i_arr], dI_dxq, vis_pmf[i_arr], 7);
 
  
}


__global__ void cu_edge_gradient_score_final(double *edge_score, double *G_edge, int num_samples, int n_edge, scope_params_t *params, int scope_round) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  double w = (scope_round==2 ? params->score2_edge_weight : params->score3_edge_weight);    
  cu_mult(&(G_edge[7*i]), &(G_edge[7*i]), w, 7);
  edge_score[i] *= w;
}

void edge_score_gradient(double *cu_edge_gradient_score, double *cu_G_edge, double *cu_samples_x, double *cu_samples_q, double *cu_P2, double *cu_P, int num_samples, int n_edge, int *cu_n, 
			 cu_double_matrix_t range_image, cu_range_image_data_t range_image_data, cu_double_matrix_t edge_image, scope_params_t *cu_params, scope_params_t *params, int scope_round, 
			 dim3 block, dim3 thread, dim3 block_size_small, dim3 thread_size_small) {

  cu_transform_cloud<<<block, thread>>>(cu_P2, cu_P, cu_samples_x, cu_samples_q, num_samples, n_edge, cu_n);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "cloud transform!\n" );


  int *cu_xi;
  cu_malloc(&cu_xi, num_samples*n_edge * sizeof(int), "xi");
  int *cu_yi;
  cu_malloc(&cu_yi, num_samples*n_edge * sizeof(int), "yi");
  cu_populate_xi_yi<<<block, thread>>>(cu_xi, cu_yi, cu_P2, range_image_data, num_samples, n_edge, cu_n);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "populate xi yi!\n" );

  
  // compute visibility of sampled model edges
  int vis_pixel_radius = 2;

  double *cu_vis_prob;
  cu_malloc(&cu_vis_prob, num_samples * n_edge * sizeof(double), "vis_prob");
  double *cu_vis_prob_sums;
  cu_malloc(&cu_vis_prob_sums, num_samples * sizeof(double), "vis_prob_sums");
  double *cu_vis_pmf;
  cu_malloc(&cu_vis_pmf, num_samples * n_edge * sizeof(double), "vis_pmf");
  get_vis_prob_sums_and_pmf(cu_vis_prob, cu_vis_prob_sums, cu_vis_pmf, cu_P2, NULL, cu_xi, cu_yi, range_image, range_image_data, vis_pixel_radius, num_samples, n_edge, cu_n, params, block, thread, 
			    block_size_small, thread_size_small, 1);
  
  double *cu_edge_gradient_score_individual;
  double *cu_dI_dxq_individual;
  cu_malloc(&cu_edge_gradient_score_individual, num_samples * n_edge * sizeof(double), "edge_gradient_score_individual");
  cu_malloc(&cu_dI_dxq_individual, num_samples * n_edge * 7 * sizeof(double), "dI_dxq_individual");  

  cu_edge_score_gradient_individual<<<block, thread>>>(cu_edge_gradient_score_individual, cu_dI_dxq_individual, cu_vis_prob, cu_vis_pmf, cu_xi, cu_yi, cu_P, cu_samples_x, cu_samples_q, edge_image, 
						    range_image_data, range_image, num_samples, n_edge, cu_n);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "edge score gradient!\n" );

  // sum up edge_scores and gradients
  cu_add_matrix_rows_slow<<<block_size_small, thread_size_small>>>(cu_edge_gradient_score, cu_edge_gradient_score_individual, num_samples, n_edge, cu_n);
  cu_add_matrix_3d_slow<<<block_size_small, thread_size_small>>>(cu_G_edge, cu_dI_dxq_individual, num_samples, n_edge, cu_n, 7);
  
  cu_edge_gradient_score_final<<<block_size_small, thread_size_small>>>(cu_edge_gradient_score, cu_G_edge, num_samples, n_edge, cu_params, scope_round);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "edge gradient final!\n" );
  

  cu_free(cu_xi, "xi");
  cu_free(cu_yi, "yi");
  cu_free(cu_vis_prob, "vis_prob");
  cu_free(cu_vis_prob_sums, "vis_prob_sums");
  cu_free(cu_vis_pmf, "vis_pmf");
  cu_free(cu_edge_gradient_score_individual, "edge gradient score individual");
  cu_free(cu_dI_dxq_individual, "dI_dxq_individual");
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

__global__ void cu_xyzn_score_gradient_individual(double *xyzn_gradient_score_individual, double *G_xyzn_individual, double *vis_prob, double *vis_pmf, int *cu_xi, int *cu_yi, 
						  double *cloud2, double *cloud, double *normals2, double *normals, double *samples_x, double *samples_q, 
						  cu_int_matrix_t range_image_cnt, cu_double_matrix3d_t range_image_points, cu_double_matrix3d_t range_image_normals, scope_noise_model_t *noise_models,
						  int num_samples, int num_surface_points, scope_params_t *params, int score_round) {

  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= num_samples || j >= num_surface_points)
    return;

  int i_arr = i * num_surface_points + j;

  xyzn_gradient_score_individual[i_arr] = 0.0;
  int k;
  for (k = 0; k < 7; ++k)
    G_xyzn_individual[7*i_arr + k] = 0.0;

  if (vis_prob[i_arr] < .01)
    return;

  int xi = cu_xi[i_arr];
  int yi = cu_yi[i_arr];
  
  // get range image cell
  if (xi == -1 || yi == -1)
    return;
  
  double range_sigma = params->range_sigma * noise_models[i_arr].range_sigma;
  double normal_sigma = params->normal_sigma * noise_models[i_arr].normal_sigma;
  double dmax_xyz = 2*range_sigma;
  double dmax_normal = 2*normal_sigma;
  double d_xyz = dmax_xyz;
  double d_normal = dmax_normal;
  double c[4];    // range image cell plane coeffs
  
  if (range_image_cnt.ptr[xi * range_image_cnt.m + yi] > 0) {
    // get distance from model point to range image cell plane
    cu_xyzn_to_plane(c, &range_image_points.ptr[xi*range_image_points.m * range_image_points.p + yi * range_image_points.p], 
		     &range_image_normals.ptr[xi*range_image_normals.m * range_image_normals.p + yi * range_image_normals.p]);
    d_xyz = fabs(cu_dot(c, &cloud2[3*i_arr], 3) + c[3]);
    //d_xyz /= noise_models[i].range_sigma;
    d_xyz = MIN(d_xyz, dmax_xyz);
    d_normal = 1.0 - cu_dot(&normals2[3*i_arr], &range_image_normals.ptr[xi*range_image_normals.m * range_image_normals.p + yi * range_image_normals.p], 3);
    //d_normal /= noise_models[i].normal_sigma;
    d_normal = MIN(d_normal, dmax_normal);
  }
  double xyz_weight = (score_round == 2 ? params->score2_xyz_weight : params->score3_xyz_weight);
  double normal_weight = (score_round == 2 ? params->score2_normal_weight : params->score3_normal_weight);
  xyzn_gradient_score_individual[i_arr] += xyz_weight * vis_pmf[i_arr] * log(cu_normpdf(d_xyz, 0, range_sigma));
  xyzn_gradient_score_individual[i_arr] += normal_weight * vis_pmf[i_arr] * log(cu_normpdf(d_normal, 0, normal_sigma));
    
  // get gradient of this point's xyz score w.r.t. model pose (x,q)
  if (d_xyz < dmax_xyz) {
    double dp_dq[3][4];
    cu_point_rotation_jacobian(dp_dq, &(samples_q[4*i]), &cloud[3*i_arr]);
    double df_dp[3];
    df_dp[0] = c[0]; df_dp[1] = c[1]; df_dp[2] = c[2];
    //double rs = range_sigma * noise_models[i].range_sigma;
    cu_mult(df_dp, df_dp, -(c[3] + cu_dot(&cloud2[3*i_arr], c, 3)) / (range_sigma*range_sigma), 3);
    G_xyzn_individual[7*i_arr + 0] = df_dp[0]; G_xyzn_individual[7*i_arr + 1] = df_dp[1]; G_xyzn_individual[7*i_arr + 2] = df_dp[2];
    cu_vec_matrix_mult_4(&G_xyzn_individual[7*i_arr + 3], df_dp, dp_dq, 3);
    cu_mult(&G_xyzn_individual[7*i_arr], &G_xyzn_individual[7*i_arr], xyz_weight * vis_pmf[i_arr], 7);
  }
  
  // get gradient of this point's normal score w.r.t. model pose (x,q)
  if (d_normal < dmax_normal) {
    double dpn_dq[3][4];
    cu_point_rotation_jacobian(dpn_dq, &(samples_q[4*i]), &normals[3*i_arr]);
    double df_dpn[3];
    df_dpn[0] = c[0]; df_dpn[1] = c[1]; df_dpn[2] = c[2];
    //double ns = normal_sigma * noise_models[i].normal_sigma;
    cu_mult(df_dpn, df_dpn, (1 - cu_dot(&normals2[3*i_arr], c, 3)) / (normal_sigma*normal_sigma), 3);
    double G_normal[7] = {0,0,0,0,0,0,0};
    cu_vec_matrix_mult_4(&G_normal[3], df_dpn, dpn_dq, 3);
    cu_mult(G_normal, G_normal, normal_weight * vis_pmf[i_arr], 7);
    cu_add(&G_xyzn_individual[7*i_arr], &G_xyzn_individual[7*i_arr], G_normal, 7);
  }
}

__global__ void cu_matrix_add(double *cu_G, double *cu_G_edge, double *cu_G_xyzn, int num_samples, int m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  for (int k = 0; k < m; ++k) {
    cu_G[m * i + k] = cu_G_edge[m * i + k] + cu_G_xyzn[m * i + k];
  }
}

__global__ void cu_init_arr(double *cu_best_score, double init_val, int num_samples) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_samples)
    return;

  cu_best_score[i] = init_val;
}

void xyzn_score_gradient(double *cu_xyzn_gradient_score, double *cu_G_xyzn, double *cu_samples_x, double *cu_samples_q, double *cu_cloud2, double *cu_cloud, double *cu_normals2, double *cu_normals, 
			 int transform, scope_noise_model_t *cu_noise_models, cu_range_image_data_t range_image_data, cu_double_matrix_t range_image, cu_int_matrix_t range_image_cnt, 
			 cu_double_matrix3d_t range_image_points, cu_double_matrix3d_t range_image_normals, 
			 int num_samples, int num_surface_points, scope_params_t *cu_params, scope_params_t *params, int score_round, 
			 dim3 block_size, dim3 threads_per_block, dim3 block_size_small, dim3 thread_size_small, dim3 block_size_sum, dim3 thread_size_sum) {
  
  if (transform) {
    cu_transform_cloud<<<block_size, threads_per_block>>>(cu_cloud2, cu_cloud, cu_samples_x, cu_samples_q, num_samples, num_surface_points, NULL);
    cu_transform_cloud<<<block_size, threads_per_block>>>(cu_normals2, cu_normals, NULL, cu_samples_q, num_samples, num_surface_points, NULL);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "transform cloud!\n" );
  }
  
  int *cu_xi;
  cu_malloc(&cu_xi, num_samples*num_surface_points * sizeof(int), "xi");
  int *cu_yi;
  cu_malloc(&cu_yi, num_samples*num_surface_points * sizeof(int), "yi");
  cu_populate_xi_yi<<<block_size, threads_per_block>>>(cu_xi, cu_yi, cu_cloud2, range_image_data, num_samples, num_surface_points, NULL);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "populate xi yi xyzn gradient!\n" );


  double *cu_vis_prob;
  cu_malloc(&cu_vis_prob, num_samples * num_surface_points * sizeof(double), "vis_prob");
  double *cu_vis_prob_sums;
  cu_malloc(&cu_vis_prob_sums, num_samples * sizeof(double), "vis_prob_sums");
  double *cu_vis_pmf;
  cu_malloc(&cu_vis_pmf, num_samples * num_surface_points * sizeof(double), "vis_pmf");

  get_vis_prob_sums_and_pmf(cu_vis_prob, cu_vis_prob_sums, cu_vis_pmf, cu_cloud2, cu_normals2, cu_xi, cu_yi, range_image, range_image_data, 0, num_samples, num_surface_points, NULL, params, 
			    block_size, threads_per_block, block_size_sum, thread_size_sum, 0);

  double *cu_xyzn_gradient_score_individual;
  cu_malloc(&cu_xyzn_gradient_score_individual, num_samples * num_surface_points * sizeof(double), "xyzn_score_gradient_individual");
  double *cu_G_xyzn_individual;
  cu_malloc(&cu_G_xyzn_individual, num_samples * num_surface_points * 7 * sizeof(double), "G_xyzn_individual");  
                    
  cu_xyzn_score_gradient_individual<<<block_size, threads_per_block>>>(cu_xyzn_gradient_score_individual, cu_G_xyzn_individual, cu_vis_prob, cu_vis_pmf, cu_xi, cu_yi, cu_cloud2, cu_cloud, cu_normals2, 
								       cu_normals, cu_samples_x, cu_samples_q, range_image_cnt, range_image_points, range_image_normals, cu_noise_models, 
								       num_samples, num_surface_points, cu_params, score_round);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "xyzn gradient individual!\n" );

  // sum up xyzn_scores and gradients
  cu_add_matrix_rows_slow<<<block_size_small, thread_size_small>>>(cu_xyzn_gradient_score, cu_xyzn_gradient_score_individual, num_samples, num_surface_points, NULL);
  cu_add_matrix_3d_slow<<<block_size_small, thread_size_small>>>(cu_G_xyzn, cu_G_xyzn_individual, num_samples, num_surface_points, NULL, 7);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "add mat 3d slow!\n" );

  
  cu_free(cu_xi, "xi");
  cu_free(cu_yi, "yi");
  cu_free(cu_vis_prob, "vis_prob");
  cu_free(cu_vis_prob_sums, "vis_prob_sums");
  cu_free(cu_vis_pmf, "vis_pmf");
  cu_free(cu_xyzn_gradient_score_individual, "edge gradient score individual");
  cu_free(cu_G_xyzn_individual, "dI_dxq_individual");
}

__global__ void cu_normalize_matrix_rows(double *cu_out, double *cu_in, int num_samples, int width) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_samples)
    return;

  cu_normalize(&(cu_out[i*width]), &(cu_in[i*width]), width);
}

__global__ void cu_mult_matrix_rows(double *cu_out, double *cu_in, double mult, double *cu_mult_arr, int num_samples, int width) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_samples)
    return;
  
  double f = mult;
  if (cu_mult_arr)
    f *= cu_mult_arr[i];

  cu_mult(&(cu_out[i * width]), &(cu_in[i*width]), f, width);
}

__global__ void cu_add_dxq(double *cu_x2, double *cu_q2, double *cu_samples_x, double *cu_samples_q, double *cu_dxq, int num_samples) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_samples)
    return;

  cu_x2[3*i] = cu_samples_x[3*i] + cu_dxq[7*i]; cu_x2[3*i + 1] = cu_samples_x[3*i + 1] + cu_dxq[7*i + 1]; cu_x2[3*i + 2] = cu_samples_x[3*i + 2] + cu_dxq[7*i + 2]; 
  cu_q2[4*i] = cu_samples_q[4*i] + cu_dxq[7*i + 3]; cu_q2[4*i + 1] = cu_samples_q[4*i + 1] + cu_dxq[7*i + 4]; 
  cu_q2[4*i + 2] = cu_samples_q[4*i + 2] + cu_dxq[7*i + 5]; cu_q2[4*i + 3] = cu_samples_q[4*i + 3] + cu_dxq[7*i + 6];
 
}

__global__ void cu_update_best(double *cu_best_score, double *cu_best_x, double *cu_best_q, double *cu_best_step, double *cu_scores, double *cu_x2, double *cu_q2, double *cu_step, double step, int num_samples) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_samples)
    return;

  if (cu_best_score[i] < cu_scores[i]) {
    cu_best_score[i] = cu_scores[i];
    cu_best_x[3*i] = cu_x2[3*i]; cu_best_x[3*i+1] = cu_x2[3*i+1]; cu_best_x[3*i+2] = cu_x2[3*i+2];
    cu_best_q[4*i] = cu_q2[4*i]; cu_best_q[4*i+1] = cu_q2[4*i+1]; cu_best_q[4*i+2] = cu_q2[4*i+2]; cu_best_q[4*i+3] = cu_q2[4*i+3];
    cu_best_step[i] = step * cu_step[i];
  }
}

void align_models_gradient(scope_sample_t *samples, int num_samples, cu_model_data_t *cu_model, cu_obs_data_t *cu_obs, scope_params_t *cu_params, scope_params_t *params, 
			   int num_points, int model_points, int round) {
  //TODO(sanja): These dim3s need some serious reorganizing/renaming
  int num_surface_points = (num_points > 0 ? num_points : model_points);

  dim3 threads_per_block(256, 1, 1);
  dim3 block_size(ceil(1.0 * num_surface_points / threads_per_block.x), num_samples);

  dim3 thread_size_small(64);
  dim3 block_size_small(ceil(1.0 * num_samples/thread_size_small.x));

  dim3 thread_size_sum(256);
  dim3 block_size_sum(1, num_samples);
  dim3 thread_size_sum_small(64);

  //TODO: make these params
  int max_iter = 20;
  
  // unpack args
  double *cu_samples_x;
  cu_malloc(&cu_samples_x, num_samples * 3 * sizeof(double), "samples_x");
  double *cu_samples_q;
  cu_malloc(&cu_samples_q, num_samples * 4 * sizeof(double), "samples_y");
  unpack_x_q(cu_samples_x, cu_samples_q, samples, num_samples);

  int n_edge = cu_model->max_num_edges;
  int *cu_n;
  cu_malloc(&cu_n, num_samples * sizeof(int), "n");
  dim3 block_size_n_edge(ceil(1.0 * n_edge / thread_size_sum.x), num_samples);
  double *cu_P;
  cu_malloc(&cu_P, num_samples * n_edge * 3*sizeof(double), "cu_P");
  int *cu_vi;
  cu_malloc(&cu_vi, num_samples * sizeof(int), "vi");
  get_range_edge_points(cu_n, cu_P, num_samples, n_edge, cu_n, num_points, cu_model, cu_vi, 1, cu_samples_x, cu_samples_q, block_size_n_edge, thread_size_sum, block_size_small, thread_size_small);
  
  double *cu_P2;
  cu_malloc(&cu_P2, num_samples * n_edge * 3*sizeof(double), "cu_P2");
  
  // get model surface points
  int *cu_idx;
  cu_malloc(&cu_idx, num_samples * num_surface_points * sizeof(int), "idxs");
  get_validation_points(cu_idx, model_points, num_surface_points, num_samples, block_size, threads_per_block);

  double *cu_cloud, *cu_cloud2, *cu_normals, *cu_normals2;
  cu_malloc(&cu_cloud, num_samples * num_surface_points * 3 * sizeof(double), "cloud");
  cu_malloc(&cu_cloud2, num_samples * num_surface_points * 3 * sizeof(double), "cloud2");
  cu_malloc(&cu_normals, num_samples * num_surface_points * 3 * sizeof(double), "normals");
  cu_malloc(&cu_normals2, num_samples * num_surface_points * 3 * sizeof(double), "normals2");

  cu_reorder_rows<<<block_size, threads_per_block>>>(cu_cloud, cu_model->points, cu_idx, num_samples, num_surface_points, 3, NULL);
  cu_reorder_rows<<<block_size, threads_per_block>>>(cu_normals, cu_model->normals, cu_idx, num_samples, num_surface_points, 3, NULL);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "reorder rows!\n" );
  
  scope_noise_model_t *cu_noise_models;
  cu_malloc(&cu_noise_models, num_samples * num_surface_points * sizeof(scope_noise_model_t), "noise_models");
  
  double *cu_G_edge, *cu_G_xyzn, *cu_G;
  cu_malloc(&cu_G_edge, 7*num_samples * sizeof(double), "G_edge");
  cu_malloc(&cu_G_xyzn, 7*num_samples * sizeof(double), "G_xyzn");
  cu_malloc(&cu_G, 7*num_samples * sizeof(double), "G");

  double *cu_edge_gradient_score;
  cu_malloc(&cu_edge_gradient_score, num_samples * sizeof(double), "edge_gradient_score");
  double *cu_xyzn_gradient_score;
  cu_malloc(&cu_xyzn_gradient_score, num_samples * sizeof(double), "edge_gradient_score");

  int *cu_xi;
  cu_malloc(&cu_xi, num_samples*num_surface_points * sizeof(int), "xi");
  int *cu_yi;
  cu_malloc(&cu_yi, num_samples*num_surface_points * sizeof(int), "yi");
  double *cu_vis_prob;
  cu_malloc(&cu_vis_prob, num_samples * num_surface_points * sizeof(double), "vis_prob");
  double *cu_vis_prob_sums;
  cu_malloc(&cu_vis_prob_sums, num_samples * sizeof(double), "vis_prob_sums");
  double *cu_vis_pmf;
  cu_malloc(&cu_vis_pmf, num_samples * num_surface_points * sizeof(double), "vis_pmf");

  double *cu_edge_scores, *cu_xyz_scores, *cu_normal_scores, *cu_scores;
  cu_malloc(&cu_edge_scores, num_samples * sizeof(double), "edge_scores");
  cu_malloc(&cu_xyz_scores, num_samples * sizeof(double), "xyz_scores");
  cu_malloc(&cu_normal_scores, num_samples * sizeof(double), "normal_scores");
  cu_malloc(&cu_scores, num_samples * sizeof(double), "scores");

  double *cu_x2, *cu_q2, *cu_dxq;
  cu_malloc(&cu_x2, num_samples * 3 * sizeof(double), "x2");
  cu_malloc(&cu_q2, num_samples * 4 * sizeof(double), "q2");
  cu_malloc(&cu_dxq, num_samples * 7 * sizeof(double), "dxq");

  double step = .01;  // step size in gradient ascent
  double step_mult[3] = {.6, 1, 1.6};

  double *cu_step;
  cu_malloc(&cu_step, num_samples * sizeof(double), "step");
  cu_init_arr<<<block_size_small, thread_size_small>>>(cu_step, step, num_samples);


  double *cu_best_score, *cu_best_step;
  cu_malloc(&cu_best_score, num_samples * sizeof(double), "best_score");
  cu_malloc(&cu_best_step, num_samples * sizeof(double), "best_step");
  double *cu_best_x, *cu_best_q;
  cu_malloc(&cu_best_x, num_samples * 3 * sizeof(double), "best_x");
  cu_malloc(&cu_best_q, num_samples * 4 * sizeof(double), "best_q");

  int j, iter;
  double init_val = -10000000.0;
  
  for (iter = 0; iter < max_iter; iter++) {
    cu_transform_cloud<<<block_size, threads_per_block>>>(cu_cloud2, cu_cloud, cu_samples_x, cu_samples_q, num_samples, num_surface_points, NULL);
    cu_transform_cloud<<<block_size, threads_per_block>>>(cu_normals2, cu_normals, NULL, cu_samples_q, num_samples, num_surface_points, NULL);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "transform cloud!\n" );

    cu_get_viewpoints<<<block_size_small, thread_size_small>>>(cu_vi, num_samples, cu_samples_x, cu_samples_q, cu_model->range_edges_model_views);

    cu_get_noise_models<<<block_size, threads_per_block>>>(cu_noise_models, cu_cloud2, cu_normals2, cu_idx, cu_vi, cu_model->ved, cu_model->normalvar, num_samples, num_surface_points);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "get noise models!\n" );

    cudaMemset(cu_G_edge, 0, 7*num_samples * sizeof(double));
    cudaMemset(cu_G_xyzn, 0, 7*num_samples * sizeof(double));
    
    edge_score_gradient(cu_edge_gradient_score, cu_G_edge, cu_samples_x, cu_samples_q, cu_P2, cu_P, num_samples, n_edge, cu_n, cu_obs->range_image, cu_obs->range_image_data, cu_obs->edge_image, 
			cu_params, params, round, block_size_n_edge, thread_size_sum, block_size_small, thread_size_small);

    xyzn_score_gradient(cu_xyzn_gradient_score, cu_G_xyzn, cu_samples_x, cu_samples_q, cu_cloud2, cu_cloud, cu_normals2, cu_normals, 0, cu_noise_models, cu_obs->range_image_data, cu_obs->range_image,
			cu_obs->range_image_cnt, cu_obs->range_image_points, cu_obs->range_image_normals, num_samples, num_surface_points, cu_params, params, round, 
			block_size, threads_per_block, block_size_small, thread_size_small, block_size_sum, thread_size_sum);
    cu_matrix_add<<<block_size_small, thread_size_small>>>(cu_G, cu_G_edge, cu_G_xyzn, num_samples, 7);

    cu_init_arr<<<block_size_small, thread_size_small>>>(cu_best_score, init_val, num_samples);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "init arr!\n" );

  
    cudaMemset(cu_best_step, 0, num_samples * sizeof(double));
  
    for (j = 0; j < 3; ++j) {
      // take a step in the direction of the gradient
      cu_normalize_matrix_rows<<<block_size_small, thread_size_small>>>(cu_G, cu_G, num_samples, 7);
      cu_mult_matrix_rows<<<block_size_small, thread_size_small>>>(cu_dxq, cu_G, step_mult[j], cu_step, num_samples, 7);
      cu_add_dxq<<<block_size_small, thread_size_small>>>(cu_x2, cu_q2, cu_samples_x, cu_samples_q, cu_dxq, num_samples);
      
      cu_normalize_matrix_rows<<<block_size_small, thread_size_small>>>(cu_q2, cu_q2, num_samples, 4);
      if ( cudaSuccess != cudaGetLastError() )
	printf( "prep stuff for step direction!\n" );

      cu_transform_cloud<<<block_size, threads_per_block>>>(cu_cloud2, cu_cloud, cu_x2, cu_q2, num_samples, num_surface_points, NULL);
      cu_transform_cloud<<<block_size, threads_per_block>>>(cu_normals2, cu_normals, NULL, cu_q2, num_samples, num_surface_points, NULL);
      if ( cudaSuccess != cudaGetLastError() )
	printf( "transform clouds!\n" );

      
      cu_populate_xi_yi<<<block_size, threads_per_block>>>(cu_xi, cu_yi, cu_cloud2, cu_obs->range_image_data, num_samples, num_surface_points, NULL);
      if ( cudaSuccess != cudaGetLastError() )
	printf( "populate xi yi!\n" );

      get_vis_prob_sums_and_pmf(cu_vis_prob, cu_vis_prob_sums, cu_vis_pmf, cu_cloud2, cu_normals2, cu_xi, cu_yi, cu_obs->range_image, cu_obs->range_image_data, 0, num_samples, num_surface_points, NULL, params, 
			    block_size, threads_per_block, block_size_sum, thread_size_sum, 0);

      // transform edge points
      cu_transform_cloud<<<block_size_n_edge, thread_size_sum>>>(cu_P2, cu_P, cu_x2, cu_q2, num_samples, n_edge, cu_n);
      if ( cudaSuccess != cudaGetLastError() )
	printf( "transform edge!\n" );
	      
      // evaluate the score
      compute_edge_score(cu_edge_scores, cu_P2, cu_obs->range_image_data, cu_obs->range_image, num_samples, n_edge, cu_n, cu_obs->edge_image, cu_model->score_comp_models->b_edge, 
			 cu_model->score_comp_models->b_edge_occ, params, cu_params, round, block_size_n_edge, block_size_sum, thread_size_sum, block_size_small, thread_size_small);

      compute_xyz_score(cu_xyz_scores, cu_cloud2, cu_xi, cu_yi, cu_vis_pmf, cu_noise_models, cu_obs->range_image, cu_obs->range_image_data, cu_obs->range_image_cnt, num_samples, num_surface_points,
			cu_params, round, cu_model->score_comp_models->b_xyz, block_size, threads_per_block, block_size_sum, thread_size_sum, block_size_small, thread_size_small);
      
      compute_normal_score(cu_normal_scores, cu_normals2, cu_vis_pmf, cu_noise_models, num_samples, num_surface_points, cu_xi, cu_yi, cu_obs->range_image_cnt, cu_obs->range_image_normals, 
			   cu_model->score_comp_models->b_normal, cu_params, round, block_size, threads_per_block, block_size_sum, thread_size_sum, block_size_small, thread_size_small);
      cu_add_3_scores<<<block_size_small, thread_size_small>>>(cu_scores, cu_xyz_scores, cu_normal_scores, cu_edge_scores, num_samples);
      if ( cudaSuccess != cudaGetLastError() )
	printf( "add 3 scores!\n" );
      cu_update_best<<<block_size_small, thread_size_small>>>(cu_best_score, cu_best_x, cu_best_q, cu_best_step, cu_scores, cu_x2, cu_q2, cu_step, step_mult[j], num_samples);
      if ( cudaSuccess != cudaGetLastError() )
	printf( "update best!\n" );
    }
    
    cudaMemcpy(cu_samples_x, cu_best_x, 3*num_samples*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(cu_samples_q, cu_best_q, 4*num_samples*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(cu_step, cu_best_step, num_samples * sizeof(double), cudaMemcpyDeviceToDevice); // NOTE(sanja): According to StackOverflow, might be expensive
    
  }
  
  double **samples_x = new_matrix2(num_samples, 3);  
  double **samples_q = new_matrix2(num_samples, 4);
  cudaMemcpy(samples_x[0], cu_samples_x, 3 * num_samples * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(samples_q[0], cu_samples_q, 4 * num_samples * sizeof(double), cudaMemcpyDeviceToHost);
  
  int i;
  for (i = 0; i < num_samples; ++i) {
    memcpy(samples[i].x, samples_x[i], 3 * sizeof(double));
  }
  for (i = 0; i < num_samples; ++i) {
    memcpy(samples[i].q, samples_q[i], 4 * sizeof(double));
  }  
 
  free_matrix2(samples_x);
  free_matrix2(samples_q);

  cu_free(cu_samples_x, "x free");
  cu_free(cu_samples_q, "q free");
  cu_free(cu_n, "n free");
  cu_free(cu_P, "P free");
  cu_free(cu_vi, "vi free");
  cu_free(cu_P2, "P2 free");
  cu_free(cu_idx, "idx free");
  cu_free(cu_cloud, "cloud free");
  cu_free(cu_cloud2, "cloud2 free");
  cu_free(cu_normals, "normals free");
  cu_free(cu_normals2, "normals2 free");
  cu_free(cu_G_edge, "G_edge free");
  cu_free(cu_G_xyzn, "G_xyzn free");
  cu_free(cu_G, "G free");
  cu_free(cu_edge_gradient_score, "edge_gradient_score");
  cu_free(cu_xyzn_gradient_score, "edge_gradient_score");
  cu_free(cu_xi, "xi");
  cu_free(cu_yi, "yi");
  cu_free(cu_vis_prob, "vis_prob");
  cu_free(cu_vis_prob_sums, "vis_prob_sums");
  cu_free(cu_vis_pmf, "vis_pmf");
  cu_free(cu_edge_scores, "edge_scores");
  cu_free(cu_xyz_scores, "xyz_scores");
  cu_free(cu_normal_scores, "normal_scores");
  cu_free(cu_scores, "scores");
  cu_free(cu_best_score, "best_score");
  cu_free(cu_best_step, "best_step");
  cu_free(cu_best_x, "best_x");
  cu_free(cu_best_q, "best_q");
  cu_free(cu_step, "step");
}
