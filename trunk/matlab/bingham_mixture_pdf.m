function p = bingham_mixture_pdf(x, B, W)
% p = bingham_mixture_pdf(x, B, W)

p = 0;
for i=1:length(B)
   p = p + W(i)*bingham_pdf(x, B(i));
end
