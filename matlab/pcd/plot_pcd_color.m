function plot_pcd_color(pcd, grayscale, colormap)

if nargin < 2
    grayscale = 0;
end

if nargin < 3
    colormap = repmat((0:256)'/256, [1,3]);
end

% cluster points into color buckets
n = length(pcd.X);

plot3(pcd.X(1), pcd.Y(1), pcd.Z(1), '.');
hold on;

if grayscale
    if grayscale==1
        BW = round((pcd.R + pcd.G + pcd.B)/3);
    elseif grayscale==2
        BW = pcd.R;
    elseif grayscale==3
        BW = pcd.G;
    else %if grayscale==4
        BW = pcd.B;
    end
    for i=0:256
        mask = (BW==i);
        if max(mask)>0
            plot3(pcd.X(mask), pcd.Y(mask), pcd.Z(mask), '.', 'Color', colormap(i+1,:));
        end
    end
else
    for i=1:n
        plot3(pcd.X(i), pcd.Y(i), pcd.Z(i), '.', 'Color', [pcd.R(i), pcd.G(i), pcd.B(i)]/256);
    end
end

hold off;
axis vis3d
axis equal
set(gca, 'Color', [0,0,0]);

