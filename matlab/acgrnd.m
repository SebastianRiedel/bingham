function x = acgrnd(S)
%x = acgrnd(S)

d = size(S,1);
x = mvnrnd(zeros(1,d), S);
x = x/norm(x);
