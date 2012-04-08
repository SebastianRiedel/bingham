/*mex_sample.c*/

/*usage
 *
 *x=mex_sample(V,Z,number) returns a numberX4 array of unit quaternions
 *sampled from a bingham distribution with principal directionc V and 
 *concentration parameters Z
 *
 *WARNING there is no check on proper V and Z values, bad data may cause a
 *segmentation fault.
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
	
	if (nrhs != 3){
		mexErrMsgTxt("Error 3 inputs expected");
	}
	
	
	int i, j, n, m;
	int num_samples;
	double Z[4];
	double *input1, *input2, *input3, *raw,*output;
	double **X;
	bingham_t B;
	
	n = mxGetM(prhs[0]);	
	m = mxGetN(prhs[0]);
    /*printf("%d %d\n",n,m);*/  
    input1 = mxGetPr(prhs[0]);

    double V[3][4];
    
	V[0][0]=input1[0];
	V[0][1]=input1[1];
	V[0][2]=input1[2];
	V[0][3]=input1[3];
	V[1][0]=input1[4];
	V[1][1]=input1[5];
	V[1][2]=input1[6];
	V[1][3]=input1[7];
	V[2][0]=input1[8];
	V[2][1]=input1[9];
	V[2][2]=input1[10];
	V[2][3]=input1[11];
	
	/*
	for (i=0;i<3;i++){
		for (j=0;j<4;j++){
			printf("%f ",V[i][j]);
		} printf("\n");
	}
	*/
	
	
    double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};	
	
	
	
	input2=mxGetPr(prhs[1]);
	for (i=0;i<3;i++){
		Z[i]=input2[i];
	}
	/*
	printf("%f %f %f\n",Z[0],Z[1],Z[2]);
	*/
	input3=mxGetPr(prhs[2]);
	
	num_samples=input3[0];
	
	bingham_new(&B, n, Vp, Z);
	
	X = new_matrix2(num_samples, n);
	bingham_sample(X, &B, num_samples);
	
	
	
		plhs[0] = mxCreateDoubleMatrix(num_samples,n, mxREAL);
		output = mxGetPr(plhs[0]);
				
		for (i=0;i<num_samples;i++){
			for (j=0;j<n;j++){
				output[i+j*num_samples]=X[i][j];
		
			}
		}
	/*
	bingham_fit(&Bf, X, num_samples, n);
	printf("%f %f %f\n", Bf.Z[0],Bf.Z[1],Bf.Z[2]);
	*/ 
	
}
