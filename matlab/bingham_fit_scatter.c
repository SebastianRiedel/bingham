

#define printf mexPrintf

#include <string.h>
#include <bingham.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int i, j, n, d;
  double *S_raw, **S, *V, *Z;
  bingham_t B;

  if (nrhs != 1)
    mexErrMsgTxt("usage: function [V Z F] = bingham_fit_scatter(S) expects one argument.");

  S_raw = mxGetPr(prhs[0]);
  d = mxGetM(prhs[0]);  /* rows */

  if (d > 4)
    mexErrMsgTxt("S must be a DxD matrix, with D<=4");

  S = (double **)mxMalloc(d*sizeof(double *));
  for (i = 0; i < d; i++)
    S[i] = S_raw + d*i;

  bingham_fit_scatter(&B, S, d);

  /* return V */
  if (nlhs > 0) {
    plhs[0] = mxCreateDoubleMatrix(d, d-1, mxREAL);
    V = mxGetPr(plhs[0]);
    memcpy(V, B.V[0], d*(d-1)*sizeof(double));
  }

  /* return Z */
  if (nlhs > 1) {
    plhs[1] = mxCreateDoubleMatrix(d-1, 1, mxREAL);
    Z = mxGetPr(plhs[1]);
    memcpy(Z, B.Z, (d-1)*sizeof(double));
  }

  /* return F */
  if (nlhs > 2) {
    plhs[2] = mxCreateDoubleScalar(B.F);
  }
}

