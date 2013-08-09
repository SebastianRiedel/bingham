B.d = 3;
B.Z = [-3,-30];
B.V = [1,0,0; 0,cos(pi/4),sin(pi/4)]';
[B.F B.df] = bingham_F(B.Z);
P = [];
for xi=1:221
    for yi=1:221
        x=(xi-111)/100;
        y=(yi-111)/100;
        z = sqrt(1-x^2-y^2);
        if isreal(z)
            P(xi,yi) = bingham_pdf([x,y,z],B);
        else
            P(xi,yi) = nan;
        end
    end
end
[X,Y] = meshgrid(-1.1:.01:1.1, -1.1:.01:1.1);
contourf(X,Y,-P',10,'LineStyle','none');
axis equal
axis off
%axis([-1.1,1.1,-1.1,1.1]);
%xlabel('x');
%ylabel('y');
colormap bone
hold on
plot(cos(0:.1:2*pi+.05), sin(0:.1:2*pi+.05), 'k-', 'LineWidth', 2);
hold off


