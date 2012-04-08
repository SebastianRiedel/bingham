function x = surface_area_hypersphere(d)
% computes the surface area of the hypersphere S^d (in R^{d+1}).

if d==0
   x = 2;
elseif d==1
   x = 2*pi;
elseif d==2
   x = 4*pi;
elseif d==3
   x = 2*pi*pi;
else
   x = (2*pi/(d-1))*surface_area_hypersphere(d-2);
end
