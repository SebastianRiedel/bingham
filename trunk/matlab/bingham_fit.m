function B = bingham_fit(X)
%B = bingham_fit(X) -- Fits a bingham distribution to a data set of unit vectors
%(in the rows of X)

n = size(X,1);
S = X'*X/n;
B = bingham_fit_scatter(S);
