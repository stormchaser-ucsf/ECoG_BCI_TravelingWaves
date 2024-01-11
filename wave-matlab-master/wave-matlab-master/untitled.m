clc;clear

addpath '/Users/runfeng/Desktop/research/project/wave-matlab-master/analysis'
addpath('/Users/runfeng/Desktop/research/project/wave-matlab-master/plotting')

% get all the files on this particular session for target 1
filepath = '/Users/runfeng/Desktop/research/data/B1/Robot3DArrow/110945/BCI_Fixed';
D=dir(filepath);
files={};
for j=3:length(D)
    filename = fullfile(filepath,D(j).name);
    load(filename)
    if TrialData.TargetID==1 || TrialData.TargetID==7 || TrialData.TargetID==3
        files=[files;filename];
    end
end
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',18,'HalfPowerFrequency2',20, ...
    'SampleRate',1e3);
bpFilt2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);
col={'k','r','g'};
pow_stat2=[];
pow_stat3=[];
for i=3:length(files)
    load(files{i})
    raw_data=TrialData.BroadbandData(1:end);
    raw_data = cell2mat(raw_data');
    chmap = TrialData.Params.ChMap;
    raw_data = filtfilt(bpFilt,raw_data);
    %raw_data = abs(hilbert(filtfilt(bpFilt2,raw_data)));
    %raw_data = filtfilt(bpFilt,raw_data);
    task_state = TrialData.TaskState;
    idx = [0 diff(task_state)];
    idx=find(idx>0);
    state1 = raw_data(1:1000,:);
    state2 = raw_data(1001:2000,:);
    state3 = raw_data(2001:3400,:);
    state4 = raw_data(3401:end,:);
    %[P, f ,Phi, lambda, Xhat, z0, Z,rf]=dmd_alg(state1',1e3,0,200);
    % [P1, f1 ,Phi, lambda, Xhat, z0, Z,rf]=dmd_alg(state2',1e3,0,200);
    % [P2, f2 ,Phi, lambda, Xhat, z0, Z,rf]=dmd_alg(state3',1e3,5,9);
    %     figure;plot(f,P)
    %     hold on
    %     plot(f1,P1)
    %     plot(f2,P2)
    % normalize the data at each channel to be within 0 and 1
    for j=1:size(raw_data,2)
        raw_data(:,j) = rescale(raw_data(:,j));
    end

    xf = zeros(8,16,size(raw_data,1));
    for j=1:size(raw_data,1)
        tmp=raw_data(j,:);
        xf(:,:,j) = tmp(chmap);
    end



    Fs = 1000;
    pixel_spacing = 1; %a.u.
    % z-score data
    xf = zscore_independent( xf );
    % do we need to z-score across channels?
    % --> yes, we need to perform z-score otherwise will have problem in phase
    % gradient process




    % form analytic signal
    xph = analytic_signal( xf ); % basically perform hilbert on xf

    % calculate instantaneous frequency
    [wt,signIF] = instantaneous_frequency( xph, Fs );
    % why we need to get instantaneous freq?

    % calculate phase gradient
    [pm,pd,dx,dy] = phase_gradient_complex_multiplication( xph, pixel_spacing, signIF );

    % plot resulting vector field
    plot_vector_field( exp( 1i .* pd(:,:,1200) ), 1 );


    % % plot movie
    % figure;
    % play_movie(xf)




    figure;
    for j=1000:size(raw_data,1)
        subplot(1,2,1)
        tmp=raw_data(j,:);
        imagesc(tmp(chmap))
        %caxis([0 1])
        colormap bone
        colorbar
        axis off
        textboxHandle = uicontrol('Style', 'text', 'Position', [0, 0, 200, 30]);
        newText = sprintf('Bin: %d', ceil( (j-200)/200 +1));
        set(textboxHandle, 'String', newText);

        subplot(1,2,2)
        ph = exp( 1i .* pd(:,:,j) );
        [XX,YY] = meshgrid( 1:size(ph,2), 1:size(ph,1) );
        M = real( exp( 1i * angle(ph) ) ); N = imag( exp( 1i * angle(ph) ) );

        % plotting

        ih = imagesc( angle(ph) ); cb = colorbar; axis image; caxis( [-pi pi] ); hold on;
        set( get(cb,'ylabel'), 'string', 'Direction (rad)' )
        quiver( XX, YY, M, N, 0.25, 'k', 'linewidth', 3 );
        set( gca, 'fontname', 'arial', 'fontsize', 14, 'ydir', 'normal' ); hh = gca;
        delete( ih ); delete( cb ); axis off; hold off;


        pause(0.0000001)


    end
end
% figure;plot(nanmean(pow_stat2,1))
% hold on
% plot(nanmean(pow_stat3,1))
% % filter it in the theta range and plot a movie
% %fvtool(bpFilt)