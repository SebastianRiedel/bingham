% syms w b real
% f1 = cos(b*acos(w))*sin(b*acos(w))/sin(acos(w));
% f2 = sin(b*acos(w))^2/sin(acos(w))^2;
% 
% % 4th order taylor-series expansion of f1 about w=0
% c0 = subs(f1, 'w', 0);
% c1 = subs(diff(f1,w,1), 'w', 0);
% c2 = subs(diff(f1,w,2), 'w', 0);
% c3 = subs(diff(f1,w,3), 'w', 0);
% c4 = subs(diff(f1,w,4), 'w', 0);
% g1 = c0 + c1*w + c2*w^2/2 + c3*w^3/6 + c4*w^4/24
% 
% % 3rd order taylor-series expansion of f2 about w=0
% c0 = subs(f2, 'w', 0);
% c1 = subs(diff(f2,w,1), 'w', 0);
% c2 = subs(diff(f2,w,2), 'w', 0);
% c3 = subs(diff(f2,w,3), 'w', 0);
% g2 = c0 + c1*w + c2*w^2/2 + c3*w^3/6


% g1 = @(w,b) (w^2*(cos((pi*b)/2)*sin((pi*b)/2) - 4*b^2*cos((pi*b)/2)*sin((pi*b)/2)))/2 + (w^4*(9*cos((pi*b)/2)*sin((pi*b)/2) - 40*b^2*cos((pi*b)/2)*sin((pi*b)/2) + 16*b^4*cos((pi*b)/2)*sin((pi*b)/2)))/24 + cos((pi*b)/2)*sin((pi*b)/2) - w*(b*cos((pi*b)/2)^2 - b*sin((pi*b)/2)^2) - (w^3*(4*b^3*sin((pi*b)/2)^2 + 4*b*cos((pi*b)/2)^2 - 4*b*sin((pi*b)/2)^2 - 4*b^3*cos((pi*b)/2)^2))/6
% g2 = @(w,b) (w^3*(8*b^3*cos((pi*b)/2)*sin((pi*b)/2) - 14*b*cos((pi*b)/2)*sin((pi*b)/2)))/6 + (w^2*(2*sin((pi*b)/2)^2 - 2*b^2*sin((pi*b)/2)^2 + 2*b^2*cos((pi*b)/2)^2))/2 + sin((pi*b)/2)^2 - 2*b*w*cos((pi*b)/2)*sin((pi*b)/2)
% 
% figure(2);
% for b=.1:.1:3
%     w = -sqrt(.75):.01:.99;
%     f1 = cos(b*acos(w)).*sin(b*acos(w))./sin(acos(w));
%     f2 = sin(b*acos(w)).^2./sin(acos(w)).^2;
%     f1_taylor = [];
%     f2_taylor = [];
%     for i=1:length(w)
%         f1_taylor(i) = g1(w(i),b);
%         f2_taylor(i) = g2(w(i),b);
%     end        
%     plot(w, f1, 'b.');
%     hold on
%     plot(w, f2, 'k.');
%     plot(w, f1_taylor, 'r-');
%     plot(w, f2_taylor, 'r-');
%     hold off
%     title(sprintf('b = %.2f', b));
%     input(':');
% end



%figure(2); for b=.1:.1:3, w0=[-sqrt(.75),-.8,-.75,-.7], for i=1:4, subplot(1,4,i); w = w0(i):.01:.99; f1 = cos(b*acos(w)).*sin(b*acos(w))./sin(acos(w)); f2 = sin(b*acos(w)).^2./sin(acos(w)).^2; fobj1 = fit(w',f1','poly4'); fobj2 = fit(w',f2','poly3'); plot(fobj1, w, f1); hold on, plot(fobj2, w, f2, 'k.'); plot(w, f1.*sqrt(1-w.^2), 'b--'); plot(w, f2.*(1-w.^2), 'k--'); hold off, legend off, title(sprintf('b = %.2f, w0 = %.2f', b, w(1))); end, input(':'); end


syms s1 s2 s3 c1 c2 c3 real
q = [c1, s1*c2, s1*s2*c3, s1*s2*s3];
unique(kron(kron(q,q),kron(q,q)))'



