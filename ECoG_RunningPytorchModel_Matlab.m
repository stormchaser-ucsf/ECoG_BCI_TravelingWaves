% ============================================================
% Import ONNX model from PyTorch and run inference on GPU
%
% PyTorch input shape: (N, C, D, H, W)
% MATLAB input shape:  (D, H, W, C, N)
% For this model specifically: (8, 11, 23, 2, N)
% dlarray labels: "SSSCB"
% ============================================================

cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/CNN3D/')
%% 1) Import the ONNX model
net = importNetworkFromONNX("phasewavecnn3d_stronger.onnx");

%% 2) Example PyTorch-style input
% Replace this with your real data in PyTorch ordering: (N, C, D, H, W)
Xpt = randn(100, 2, 8, 11, 23, "single");

%% 3) Reorder dimensions from PyTorch -> MATLAB
% (N, C, D, H, W) -> (D, H, W, C, N)
X = permute(Xpt, [3 4 5 2 1]);

%% 4) Move data to GPU
X = gpuArray(X);

%% 5) Wrap as dlarray with correct labels
dlX = dlarray(X, "SSSCB");

%% 6) Run inference
dlY = predict(net, dlX);

%% 7) Gather outputs back to CPU
score = gather(extractdata(dlY));   % raw logits
prob  = 1 ./ (1 + exp(-score));     % sigmoid probability
pred  = score > 0;                  % binary prediction

%% 8) Display results
disp("Score (logit):")
disp(score)

disp("Probability:")
disp(prob)

disp("Prediction:")
disp(pred)

% ============================================================
% Notes:
% - If you already have MATLAB-formatted data, skip the permute step.
% - If your data is already on the GPU, skip gpuArray(X).
% - For batch inference, use:
%       Xpt size = (N, 2, 8, 11, 23)
%   then the permute line will produce:
%       X size   = (8, 11, 23, 2, N)
% ============================================================


%%
% ============================================================
% CPU vs GPU timing comparison for PhaseWave CNN
% Batch size = 100
% ============================================================

% Enable forward compatibility for your GPU
parallel.gpu.enableCUDAForwardCompatibility(true);

% Load model
net = importNetworkFromONNX("phasewavecnn3d_stronger.onnx");

% ------------------------------------------------------------
% Create test input in PyTorch format: (N, C, D, H, W)
% ------------------------------------------------------------
N = 100;
Xpt = randn(N, 2, 8, 11, 23, "single");

% Convert to MATLAB format: (D, H, W, C, N)
X = permute(Xpt, [3 4 5 2 1]);

% ------------------------------------------------------------
% CPU version
% ------------------------------------------------------------
dlX_cpu = dlarray(X, "SSSCB");

cpuFcn = @() predict(net, dlX_cpu);

cpu_time = timeit(cpuFcn);

% ------------------------------------------------------------
% GPU version
% ------------------------------------------------------------
dlX_gpu = dlarray(gpuArray(X), "SSSCB");

% Warm-up run (important for GPU timing accuracy)
predict(net, dlX_gpu);

gpuFcn = @() predict(net, dlX_gpu);

gpu_time = gputimeit(gpuFcn);

% ------------------------------------------------------------
% Display results
% ------------------------------------------------------------
fprintf("Batch size: %d\n", N);
fprintf("CPU time: %.6f seconds\n", cpu_time);
fprintf("GPU time: %.6f seconds\n", gpu_time);
fprintf("Speedup (CPU/GPU): %.2fx\n", cpu_time / gpu_time);

%%

% ============================================================
% Find practical optimal batch size on GPU and compare CPU vs GPU
% Repeats each timing experiment 50 times and reports averages
%
% Model input:
%   PyTorch format: (N, C, D, H, W)
%   MATLAB format:  (D, H, W, C, N)
% ============================================================

clear; clc;

% 0) Setup
parallel.gpu.enableCUDAForwardCompatibility(true);

net = importNetworkFromONNX("phasewavecnn3d_stronger.onnx");
gpuDevice;  % confirm GPU is available

% Batch sizes to test
batch_sizes = [1 2 4 8 16 32 64 100 128 256 512 1024];

% Number of repeated timing measurements per batch size
nRepeats = 50;

% Storage
cpu_mean = zeros(size(batch_sizes));
cpu_std  = zeros(size(batch_sizes));
gpu_mean = zeros(size(batch_sizes));
gpu_std  = zeros(size(batch_sizes));
speedup_mean = zeros(size(batch_sizes));

% 1) Benchmark loop
for i = 1:numel(batch_sizes)
    N = batch_sizes(i);

    fprintf('\nTesting batch size %d...\n', N);

    % --------------------------------------------------------
    % Create input in PyTorch format: (N, C, D, H, W)
    % Convert to MATLAB format:      (D, H, W, C, N)
    % --------------------------------------------------------
    Xpt = randn(N, 2, 8, 11, 23, "single");
    X   = permute(Xpt, [3 4 5 2 1]);

    % CPU input
    dlX_cpu = dlarray(X, "SSSCB");

    % GPU input
    dlX_gpu = dlarray(gpuArray(X), "SSSCB");

    % Warm-up GPU once before timing
    predict(net, dlX_gpu);
    wait(gpuDevice);

    % --------------------------------------------------------
    % Repeat CPU timings
    % --------------------------------------------------------
    cpu_times = zeros(nRepeats,1);
    for r = 1:nRepeats
        t = tic;
        predict(net, dlX_cpu);
        cpu_times(r) = toc(t);
    end

    % --------------------------------------------------------
    % Repeat GPU timings
    % Must synchronize GPU each repetition
    % --------------------------------------------------------
    gpu_times = zeros(nRepeats,1);
    for r = 1:nRepeats
        wait(gpuDevice);      % ensure prior work is done
        t = tic;
        predict(net, dlX_gpu);
        wait(gpuDevice);      % ensure GPU work finishes before toc
        gpu_times(r) = toc(t);
    end

    % Save summary stats
    cpu_mean(i) = mean(cpu_times);
    cpu_std(i)  = std(cpu_times);
    gpu_mean(i) = mean(gpu_times);
    gpu_std(i)  = std(gpu_times);
    speedup_mean(i) = cpu_mean(i) / gpu_mean(i);

    fprintf('CPU mean: %.6f s | GPU mean: %.6f s | Speedup: %.2fx\n', ...
        cpu_mean(i), gpu_mean(i), speedup_mean(i));
end

% 2) Build summary table
T = table( ...
    batch_sizes(:), ...
    cpu_mean(:), cpu_std(:), ...
    gpu_mean(:), gpu_std(:), ...
    speedup_mean(:), ...
    'VariableNames', { ...
        'BatchSize', ...
        'CPU_Mean_s', 'CPU_Std_s', ...
        'GPU_Mean_s', 'GPU_Std_s', ...
        'Speedup_CPU_over_GPU'});

disp(T)

% 3) Practical "optimal" batch size choices
% A) Best raw GPU throughput = smallest time per sample
gpu_time_per_sample = gpu_mean ./ batch_sizes;
[~, idx_best_throughput] = min(gpu_time_per_sample);

best_batch_throughput = batch_sizes(idx_best_throughput);
best_gpu_mean = gpu_mean(idx_best_throughput);
best_gpu_per_sample = gpu_time_per_sample(idx_best_throughput);

% B) Best speedup over CPU
[~, idx_best_speedup] = max(speedup_mean);
best_batch_speedup = batch_sizes(idx_best_speedup);
best_speedup = speedup_mean(idx_best_speedup);

fprintf('\n====================================================\n');
fprintf('Best GPU throughput batch size: %d\n', best_batch_throughput);
fprintf('Mean GPU time at that batch size: %.6f s\n', best_gpu_mean);
fprintf('GPU time per sample there: %.6f ms/sample\n', best_gpu_per_sample*1000);
fprintf('\nBest CPU/GPU speedup batch size: %d\n', best_batch_speedup);
fprintf('Mean speedup there: %.2fx\n', best_speedup);
fprintf('====================================================\n');

%4) Optional plots
figure;
plot(batch_sizes, cpu_mean*1000, '-o', 'LineWidth', 1.5); hold on;
plot(batch_sizes, gpu_mean*1000, '-o', 'LineWidth', 1.5);
xlabel('Batch size');
ylabel('Mean inference time (ms)');
legend('CPU','GPU','Location','northwest');
title('CPU vs GPU inference time');
grid on;

figure;
plot(batch_sizes, speedup_mean, '-o', 'LineWidth', 1.5);
xlabel('Batch size');
ylabel('Speedup (CPU time / GPU time)');
title('GPU speedup over CPU');
grid on;

figure;
plot(batch_sizes, gpu_time_per_sample*1000, '-o', 'LineWidth', 1.5);
xlabel('Batch size');
ylabel('GPU time per sample (ms/sample)');
title('GPU throughput efficiency');
grid on;

