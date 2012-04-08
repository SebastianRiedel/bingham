%mex -I/u/matlab/jglov/bingham/c/include -L/u/matlab/jglov/bingham/c -lbingham -lgsl -lgslcblas bingham_fit.c
%mex -I/u/matlab/jglov/bingham/c/include -L/u/matlab/jglov/bingham/c -lbingham -lgsl -lgslcblas bingham_fit_scatter.c
%mex -I/home/jglov/bingham/c/include -L/home/jglov/bingham/c -lbingham -lgsl bingham_fit.c
%mex bingham_fit.c
%mex -L/usr/include -lbingham -lgsl bingham_fit_scatter.c
