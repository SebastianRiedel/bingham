function h = bingham_mixture_entropy(B,W)
% h = bingham_mixture_entropy(B,W)

h = 0;
for i=1:length(B)
    h = h + W(i)*bingham_entropy(B(i));
end
