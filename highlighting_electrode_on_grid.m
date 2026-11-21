% Grid dimensions
nRows = 23;
nCols = 11;

% Create grid coordinates
[X, Y] = meshgrid(1:nCols, 1:nRows);

figure('Color', 'w');
hold on;

% Plot all electrodes as black dots
scatter(X(:), Y(:), 1, 'k', 'filled');

% Bottom-left 5x5 electrodes (active channels)
% activeX = X(end-4:end, 1:5);
% activeY = Y(end-4:end, 1:5);

% Top-left 5x5 electrodes (active channels)
activeX = X(1:6, 1:6);
activeY = Y(1:6, 1:6);


% Plot active electrodes as larger red dots
scatter(activeX(:), activeY(:), 7, 'r', 'filled');

% Formatting
axis equal;
axis tight;
axis off;
set(gca, 'YDir', 'reverse');

hold off;

