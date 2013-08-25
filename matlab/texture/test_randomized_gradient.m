%test script for randomized_gradient_descent symm_bingham_fit
%requires MTEX for visualization

clear all
close all

%initialize a bingham distibution
B=struct('Z',[],'d',4,'V',[],'F',[],'dF',[]);

V=eye(4);
V(:,4)=[];

B.V=V;

Z=[-15,-15,-15];
B.Z=Z;

%sample members from B

X=bingham_sample(B,500);

%test fit
B2=bingham_fit(X);

%%
%symmetrize quaternions and select random variant
cs=symmetry('cubic');
ss=symmetry('triclinic');

x_double=X(:,[4,1,2,3]);
x=quaternion(x_double(:,1),x_double(:,2),x_double(:,3),x_double(:,4));
%plot(x);
x_sym=symmetrise(x,cs,ss)';

figure
plot(x_sym);

variant=randint(size(x,1),1,[1 size(cs,1)]);
x_rand_variant=x;

for ii=1:size(x_sym,2)
    x_rand_variant(ii)=x_sym(variant(ii),ii);
end

% figure
% plot(x_rand_variant);

x_fz= project2FundamentalRegion(x_rand_variant,cs,ss,x_rand_variant(1)); 

X_fz=squeeze(double(x_fz));
X_fz = X_fz(:, [4 1 2 3]);

%%

B_fit=symm_bingham_fit_srn(X_fz,'cubic');
%%
% Z=B_fit.Z;
% V=B_fit.V;
% % 
% [Z IX]=sort(Z);
% V=V(:,IX);
% % 
% B_fit.Z=Z;
% B_fit.V=V;
%%
X_fit=bingham_sample(B_fit,100);

%X_fit=X_fit(:,[4,1,2,3]);
x_fit=quaternion(X_fit(:,4),X_fit(:,1),X_fit(:,2),X_fit(:,3));
xx=symmetrise(x_fit,cs,ss)';
figure;plot(xx);

%%

ZZ=convert_concentration(B.Z);
VV=convert_basis(B.V);

figure
odf_base=BinghamODF(ZZ,VV,cs,ss);
plot(odf_base,'sections',6);


ZZ=convert_concentration(B_fit.Z);
VV=convert_basis(B_fit.V);
figure
odf_fit=BinghamODF(ZZ,VV,cs,ss);
plot(odf_fit,'sections',6);
%%
ebsd=EBSD(orientation(rotation(x_fz),cs,ss));
psi=calcKernel(ebsd);
odf_fourier=calcODF(ebsd,psi,'bandwidth',32);

figure
plot(odf_fourier,'sections',6);

