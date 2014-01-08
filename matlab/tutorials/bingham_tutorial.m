%% libbingham Matlab Tutorial
% 
%
%


%% What is a Bingham Distribution?
% The Bingham distribution is an antipodally-symmetric
% probability distribution on a unit hypersphere. Its probability
% density function (PDF) is
%
% $$ f(\vec{x} ; \Lambda, V) = \frac{1}{F} \exp \{ \sum_{i=1}^d \lambda_i (\vec{v_i}^T \vec{x})^2 \} $$
%
% where $\vec{x}$ is a unit vector on the surface of the sphere $\bf{S}^d \subset \bf{R}^{d+1}$,
% $F$ is a normalization constant,
% $\Lambda$ is a vector of non-positive ($\leq 0$) concentration parameters,
% and the columns of the $(d+1) \times d$ matrix $V$ are orthogonal unit vectors.
%
% Note that a large-magnitude $\lambda_i$ indicates that the distribution is
% highly peaked along the direction $\vec{v_i}$, while a small-magnitude
% $\lambda_i$ indicates that the distribution is spread out along $\vec{v_i}$.
%
% The Bingham distribution is derived from a zero-mean Gaussian on
% $\bf{R}^{d+1}$, conditioned to lie on the surface of the unit
% hypersphere $\bf{S}^d$.  Thus, the exponent of the Bingham PDF is
% the same as the exponent of a zero-mean Gaussian distribution (in
% principal components form, with one of the eigenvalues of the
% covariance matrix set to infinity).
%
% The Bingham distribution is the _maximum entropy_ distribution on the
% hypersphere which matches the sample inertia matrix  $E[\vec{x} \vec{x}^T]$.
% Therefore, it may be better suited to representing random process noise on the hypersphere than some other distributions,
% such as (projected) tangent-space Gaussians.  Binghams are also quite flexible, since a concentration parameter, $\lambda_i$, of
% zero indicates that the distribution is completely uniform in the direction of $\vec{v_i}$.  They are therefore very useful
% in tracking problems where there is high, anisotropic noise.


%% Matlab Support in libbingham
% Starting with version 0.3.0, all of the core functions in libbingham are
% supported in Matlab.  But, as in the C library, most functions only
% support Bingham distributions up to dimension 4.  (That is, up to
% $\bf{S}^3$.)


%% Creating a Bingham Distribution in Matlab
% Bingham distributions are represented as a Matlab struct, with fields
% _d_, _V_, _Z_, and _F_ and _dF_ (which are computed by libbingham).
% To create a new Bingham distribution, create a new struct with dimension
% _d_, orthogonal direction matrix _V_, and concentration parameters _Z_.
% For example, the uniform Bingham distribution on the 3-D sphere $\bf{S}^2$ is:

B = struct();
B.d = 3;
B.Z = [0,0];
B.V = [0,0; 1,0; 0,1];

%%
% To look up the normalization constant and its partial derivatives with
% respect to _Z_, use:

[B.F B.dF] = bingham_F(B.Z);


%% Fitting
% Given a matrix _X_ with unit vectors in the rows, you can compute the
% maximum likelihood Bingham distribution given _X_ with _bingham_fit()_.
% For example:

% create n 4-D unit vectors
n = 10;
X = randn(n,4);
X = X./repmat(sqrt(sum(X.^2,2)), [1,4]);

% Fit a Bingham distribution to X
B = bingham_fit(X);


%% Sampling
% To sample _n_ unit vectors from a Bingham distribution, use:

Y = bingham_sample(B,n);


%% Computing the PDF
% To compute the PDF of a unit vector _x_ under a Bingham _B_, use:

f = bingham_pdf(x,B);


%% Computing the Mode

mu = bingham_mode(B);


%% Computing the Entropy

h = bingham_entropy(B);


%% Computing the Scatter Matrix
% To compute the scatter matrix, $E[\vec{x} \vec{x}^T]$, use:

S = bingham_scatter(B);


%% Multiplying two Binghams
% Two multiply two Binghams, _B1_ and _B2_, use:

B = bingham_mult(B1,B2);


%% Special Functions for the Quaternion Bingham Distribution

B2 = bingham_pre_rotate_3d(B,q);
B2 = bingham_post_rotate_3d(q,B);
B2 = bingham_invert_3d(B);




