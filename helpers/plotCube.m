function plotCube(cx,cy,cz, width)
    % centroid = [x y z]
    % width = side length of cube

    % cx = centroid(1);
    % cy = centroid(2);
    % cz = centroid(3);

    w = width / 2;

    % Define 8 vertices of the cube
    vertices = [
        cx-w, cy-w, cz-w;
        cx+w, cy-w, cz-w;
        cx+w, cy+w, cz-w;
        cx-w, cy+w, cz-w;
        cx-w, cy-w, cz+w;
        cx+w, cy-w, cz+w;
        cx+w, cy+w, cz+w;
        cx-w, cy+w, cz+w
    ];

    % Define faces using vertex indices
    faces = [
        1 2 3 4;   % bottom
        5 6 7 8;   % top
        1 2 6 5;   % front
        2 3 7 6;   % right
        3 4 8 7;   % back
        4 1 5 8    % left
    ];

    % Plot cube
    patch('Vertices', vertices, ...
          'Faces', faces, ...
          'FaceColor', [0.2 0.6 1.0], ...
          'FaceAlpha', 0.1, ...
          'EdgeColor', 'k', ...
          'LineWidth', 1.0);

    axis equal
    grid on
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    view(3)
end