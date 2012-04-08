/*      mex_pdf.c */

/* usage p=mex_pdf(V,Z,x) returns f(x|V,Z) where x is unit quaternion
 *V is a 4X3 array of principal directions and Z is a 1X3 array of
 *concentration parameters
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bingham.h"
#include "bingham/util.h"
#include "bingham/bingham_constants_matlab.h"
#include "bingham/hypersphere.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
	int i, j, n, m;
	int num_samples;
	double Xp[4];
	double *input1, *input2, *input3, *input4, *Z, *f, *output;
	double **V, **X;
	bingham_t B,Bf;
	
	n = mxGetM(prhs[0]);	
	m = mxGetN(prhs[0]);
      
    input1 = mxGetPr(prhs[0]);

    V=new_matrix2(n,m);
    
    for (i=0;i<n;i++){
		for (j=0;j<m;j++){
			V[i][j]=input1[i+j*n];
		}
	}
	double *Vp[3] = {&V[0][0], &V[1][0], &V[2][0]};	
	Z=mxGetPr(prhs[1]);
	
	input3=mxGetPr(prhs[2]);
	
	num_samples=input3[0];
	
	input4=mxGetPr(prhs[3]);
	n = mxGetM(prhs[3]);	
	m = mxGetN(prhs[3]);
	X=new_matrix2(n,m);
    for (i=0;i<n;i++){
		for (j=0;j<m;j++){
			X[i][j]=input4[i+j*n];
		}
	}
	
	bingham_new(&B, 4, Vp, Z);

	safe_malloc(f,num_samples,double);


	for (i=0;i<num_samples;i++){
		Xp[0]=X[i][0];
		Xp[1]=X[i][1];
		Xp[2]=X[i][2];
		Xp[3]=X[i][3];
		f[i]=bingham_pdf(Xp,&B);
		/*printf("%f %f %f %f\n", Xp[0],Xp[1],Xp[2],Xp[3]);*/
	}
	/*
	for (i=0;i<num_samples;i++){
		
		printf("%f\n",f[i]);
	} */

    plhs[0] = mxCreateDoubleMatrix(num_samples,1, mxREAL);
	output = mxGetPr(plhs[0]);	
	for (i=0;i<num_samples;i++){
		output[i]=f[i];
	}

	
}
