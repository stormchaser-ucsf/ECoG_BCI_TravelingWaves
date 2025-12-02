function plot_circ_linear_regression_with_residuals(x, y, phase, a, b, theta0)
% x, y      : electrode coordinates
% phase     : electrode phase values (radians)
% a, b      : regression slopes
% theta0    : phase offset (radians)

% Ensure column vectors
x = x(:);
y = y(:);
phase = wrapTo2Pi(phase(:));

% Predicted phase from regression plane
pred_phase = wrapTo2Pi(a*x + b*y + theta0);

figure; hold on;

% -----------------------------------------------------------
% 1. Regression plane (only through electrode locations)
% -----------------------------------------------------------
tri = delaunay(x, y);

trisurf(tri, x, y, pred_phase, ...
    'FaceAlpha', 0.35, 'EdgeColor', 'none');
colormap(hsv);
shading interp;

% -----------------------------------------------------------
% 2. Actual electrode phases
% -----------------------------------------------------------
scatter3(x, y, phase, 120, phase, 'filled', ...
    'MarkerEdgeColor','k', 'LineWidth', 1);

% -----------------------------------------------------------
% 3. Plot residual lines from electrode -> plane
% -----------------------------------------------------------
for i = 1:numel(x)
    plot3([x(i) x(i)], [y(i) y(i)], ...
          [pred_phase(i) phase(i)], ...
          'k-', 'LineWidth', 1.2);
end

% -----------------------------------------------------------
% Labels and formatting
% -----------------------------------------------------------
xlabel('x (mm)');
ylabel('y (mm)');
zlabel('phase (rad)');
title('Circular–Linear Regression with Residuals');
colorbar;
view(45, 25);
grid on; axis tight;

end
