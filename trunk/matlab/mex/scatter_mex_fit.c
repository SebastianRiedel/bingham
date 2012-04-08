/*scatter_mex_fit.c*/


/* usage
 *
 *[V Z]=scatter_mex_fit(S) returns principal directions V and 
 *concentration parameters Z for a bingham distribution given 4X4 scatter 
 *matriz S
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/bingham_constants_matlab.h"
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
	if (nrhs != 1){
		mexErrMsgTxt("Error only 1 input expected");
	}
	
	if (mxGetN(prhs[0]) != 4){
		mexErrMsgTxt("Input data must be an Nx4 array");
	}
	
	
	int i, j, n, d;
	double F;
	double *input, *V, *Z, *raw;
	double **scatter;
	bingham_t B;
		
	n = mxGetM(prhs[0]);
    d = mxGetN(prhs[0]);
    /*
    mexPrintf("n=%d,   d=%d\n",n,d); 
    */
    input = mxGetPr(prhs[0]);
/*
	mexPrintf("%f %f %f %f\n",input[0],input[0+n],input[0+2*n],input[0+3*n]);
*/

    scatter=new_matrix2(n,d);
    
    for (i=0;i<n;i++){
		for (j=0;j<d;j++){
			scatter[i][j]=input[i+j*n];
		}
	}
	/*
	for (i=0;i<n;i++){
		for (j=0;j<d;j++){
			mexPrintf("%f ",scatter[i][j]);
		}
		mexPrintf("/n");
	}*/
	

	bingham_fit_scatter(&B,scatter,d);
	/*void bingham_fit_scatter(bingham_t *B, double **S, int d)*/
	
		
	free_matrix2(scatter);
	if (nlhs > 0) {
		plhs[0] = mxCreateDoubleMatrix(d, d-1, mxREAL);
		V = mxGetPr(plhs[0]);
		memcpy(V, B.V[0], d*(d-1)*sizeof(double));
	}

	if (nlhs > 1) {
		plhs[1] = mxCreateDoubleMatrix(d-1, 1, mxREAL);
		Z = mxGetPr(plhs[1]);
		memcpy(Z, B.Z, (d-1)*sizeof(double));
	}
	
	if (nlhs > 2) {
		plhs[2] = mxCreateDoubleScalar(B.F);

	}

     

	
}
