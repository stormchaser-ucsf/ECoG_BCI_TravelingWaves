
%% (MAIN): LOOKING at the the relationship b/w decoding information at each channel and
% PAC b/w hg and alpha/delta at that channel
% B3



clc;clear
close all



if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')
    load session_data_B3_Hand
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath('C:\Users\nikic\Documents\MATLAB\CircStat2012a')
    addpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers')
    

else
    %root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    root_path = '/media/user/Data/ecog_data/ECoG BCI/GangulyServer/Multistate B3/';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
end


d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);

% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
%     'SampleRate',1e3);


d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

d3 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
    'SampleRate',1e3);



imaging_B3_waves;
close all

plot_true = false;


reg_days=[];
mahab_dist_days=[];
pac_days=[];

for i=1:length(session_data)



    folders_imag =  strcmp(session_data(i).folder_type,'I');
    folders_online = strcmp(session_data(i).folder_type,'O');
    folders_batch = strcmp(session_data(i).folder_type,'B');
    folders_batch1 = strcmp(session_data(i).folder_type,'B1');
    %batch_idx_overall = [find(folders_batch==1) find(folders_batch1==1)];

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    %batch_idx = find(folders_batch==1);
    %batch_idx1 = find(folders_batch1==1);

    imag_idx = find(folders_imag==1);
    online_idx = find(folders_online==1);
    if sum(folders_batch1)==0
        batch_idx = find(folders_batch==1);
    else
        batch_idx = find(folders_batch1==1);
    end




    %     if sum(folders_batch1)==0
    %         batch_idx = find(folders_batch==1);
    %     else
    %         batch_idx = find(folders_batch1==1);
    %     end
    %     %batch_idx = [online_idx batch_idx];



    online_idx=[online_idx batch_idx];
    %batch_idx = [online_idx batch_idx_overall];


    % %%%%% get imagined data files
    % folders = session_data(i).folders(imag_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' OL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')


    %%%%%% getting online files now
    folders = session_data(i).folders(online_idx);
    day_date = session_data(i).Day;
    files=[];
    for ii=1:length(folders)
        folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
        %cd(folderpath)
        files = [files;findfiles('mat',folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' CL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %pac=zeros(100,253);
    pac_days(i,:) = abs(mean(pac,1));


    % get the mahab dist at each channel
    [mahab_dist] = get_mahab_dist(files);
    mahab_dist_days(i,:) = mahab_dist;


    if plot_true

        figure;
        % plot pac as brain image
        pac_tmp = abs(mean(pac));
        ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
            pac_tmp(116:end)];
        %figure;
        subplot(1,3,1)
        imagesc(ch_wts(ecog_grid))
        title(['PAC Day ' num2str(i)])
        ch_wts_pac=ch_wts;        

        % plot mahab dist as brain image
        ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
            mahab_dist(116:end)];
        %figure;
        subplot(1,3,2)
        imagesc(ch_wts(ecog_grid))
        title(['Mahab dist Day ' num2str(i)])
        ch_wts_mahab=ch_wts;

        % hand knob weights
        %hnd2 = ch_wts(ecog_grid(2:3,3:4));


        % plot and see?
        %figure;
        subplot(1,3,3)
        plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
        xlabel('Mahab Dist')
        ylabel('PAC')
        title(['CL Day ' num2str(i)])
        ylim([0 0.7])
        plot_beautify
        hold on

    end




    % regression
    y = abs(mean(pac))';
    x = mahab_dist';
    x = [ones(size(x,1),1) x];
    %[B,BINT,R,RINT,STATS1] = regress(y,x);
    mdl = fitlm(x(:,2),y,'RobustOpts','on');
    B = mdl.Coefficients.Estimate;
    STATS1 = [0 mdl.Coefficients.pValue'];
   

    if plot_true

        yhat = x*B;
        plot(x(:,2),yhat,'k','LineWidth',1)

        % %%plot mahab dist on brain
        
        figure;
        plot_on_brain(ch_wts_mahab,cortex,elecmatrix,ecog_grid)
          title(['hG decoding info CL Day ' num2str(i)])

        % 
        % % plot PAC on brain
        % %figure
        % plot_on_brain(ch_wts_pac,cortex,elecmatrix,ecog_grid)
        % title(['hG-delta PAC CL Day ' num2str(i)])

    end

    reg_days(:,i) = [B; STATS1(3);mdl.Coefficients.SE];

    %
    % %%%%%%%%% getting batch files now
    % folders = session_data(i).folders(batch_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' CL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    %
    % % plot pac on brain
    % pac_tmp = abs(mean(pac));
    % ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
    %     pac_tmp(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % plot mahab dist on brain grid
    % ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
    %     mahab_dist(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % hand knob weights
    % hnd9b = ch_wts(ecog_grid(2:3,3:4));
    %
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')
    % title(['CL Day ' num2str(i)])
    % ylim([0 0.7])
    % plot_beautify
    %


end

%%%%% p value trend
figure;hold on
plot(log(reg_days(3,:)),'.','MarkerSize',20)
xlabel('Days')
ylabel('Log P-value')
xticks(1:10)
y = log(reg_days(3,:))';
x = (1:10)';
x = [ones(size(x,1),1) x];
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
%plot(x(:,2),yhat,'k')
plot_beautify
title('Evolution of mu-hG PAC and discriminability (sig)')
hline(log(0.01),'--r')
xlim([0.5 10.5])


%%%% slope trend
figure;hold on
plot((reg_days(2,:)),'.','MarkerSize',20)
xlabel('Days')
ylabel('Slope between mu-hG PAC and hG Decoding')
%ylabel('LFO - hG PAC and hG decoding info.')
xticks(1:10)
y = reg_days(2,:)';
x = (1:10)';
x = [ones(size(x,1),1) x];
%[B,BINT,R,RINT,STATS1] = regress(y,x);
mdl = fitlm(x(:,2),y,'RobustOpts','on');
B = mdl.Coefficients.Estimate;
yhat = x*B;
plot(x(:,2),yhat,'k')
plot_beautify
%title('Evolution of alpha-hG PAC and discriminability')
xlim([0.5 10.5])
hline(0,'--r')

save mahab_pac_alpha_hg_B3_Hand_New -v7.3

%%%%% HIERARCHICAL LINEAR MIXED EFFECT MODEL
% change in mu-hG PAC across-days using mixed effect model

pac_days=pac_days';
mahab_days=  mahab_dist_days';

[nchan,ndays]=size(pac_days);
[chanGrid, dayGrid] = ndgrid(1:nchan, 1:ndays);
T = table;
T.Y      = mahab_days(:);         % decoding info
T.X      = pac_days(:);           % PAC
T.DayNum = dayGrid(:);             % 1..10
T.ChanID = categorical(chanGrid(:)); % same channels repeated across days
T.DayC = T.DayNum - mean(1:ndays);  % mean(1:10)=5.5

lme = fitlme(T, 'Y ~ 1 + X*DayC + (1|ChanID)', ...
    'FitMethod','REML');

% plotting
b = fixedEffects(lme);
names = lme.CoefficientNames;

bX  = b(strcmp(names,'X'));
bXD = b(strcmp(names,'X:DayC'));

days  = (1:ndays)';
dayC  = days - mean(1:ndays);
slope_hat = bX + bXD*dayC;

figure; plot(days, slope_hat, 'o-');
yline(0,'--'); xlabel('Day'); ylabel('Estimated PAC→Decoding slope');


%%%% MIXED EFFECT MODEL LOOKING AT EMERGENCE OF DECODING INFORMATION ROI
% first thing, make sure that ecog_grid is converted to 253
ecog_grid_253=[];
for i=1:size(ecog_grid,1)
    for j=1:size(ecog_grid,2)
        if ecog_grid(i,j) <= 107
            ecog_grid_253(i,j) = ecog_grid(i,j);
        elseif ecog_grid(i,j) > 108 && ecog_grid(i,j) <= 113
            ecog_grid_253(i,j) = ecog_grid(i,j)-1;

        elseif  ecog_grid(i,j) > 113 && ecog_grid(i,j) <= 118
            ecog_grid_253(i,j) = ecog_grid(i,j)-2;

        elseif ecog_grid(i,j) > 118
            ecog_grid_253(i,j) = ecog_grid(i,j)-3;
        end
    end
end

m1=[202	218	49	131	156	134
205	62	45	137	157	140
208	59	41	167	30	145
211	56	38	171	28	149
215	52	35	175	25	152
219	48	160	179	21	23];

pmv=[104	106	245	88	70
100	102	249	85	66
225	98	252	82	191
229	223	125	79	195
233	227	123	75	199
238	231	120	71	203];

lpmv1=[243	236	116	67	206
119	241	112	192	209
115	246	108	196	212
111	250	76	200	216
107	124	72	204	220
103	121	68	207	94];

spch_pmv=[99	117	193	210	90
224	113	197	213	86
228	109	201	217	83
232	105	234	221	80
237	101	239	95	77
242	97	244	91	73];

tg1=[247	222	248	87	69
122	226	251	84	65
118	230	253	81	190
114	235	96	78	194
110	240	92	74	198];

tg2=[168	187	50	136	20	1
172	63	46	142	15	126
176	60	42	147	9	132
180	57	12	151	3	138
214	53	6	154	128	143];

lm1=[93	44	163	183	16	19
89	40	166	186	10	14
55	37	170	189	4	8
51	34	174	32	129	2
47	159	178	31	135	127
43	162	182	29	141	133];

spchm1=[39	165	185	26	146	139
36	169	188	22	150	144
33	173	64	17	153	148
158	177	61	11	155	18
161	181	58	5	27	13
164	184	54	130	24	7];

nChan = 253;
ROI = strings(nChan,1);
ROI(m1(:))        = "M1";
ROI(pmv(:))       = "PMv";
ROI(lpmv1(:))     = "lPMv1";
ROI(spch_pmv(:))  = "spchPMv";
ROI(tg1(:))       = "TG1";
ROI(tg2(:))       = "TG2";
ROI(lm1(:))       = "lM1";
ROI(spchm1(:))    = "spchM1";

ROI = categorical(ROI);

%%%%  LME
DEC = mahab_days;
DEC = DEC ./ max(DEC); % normalize it to per day relative ROI

[nChan,nDay] = size(DEC);
[chanGrid, dayGrid] = ndgrid(1:nChan, 1:nDay);

T = table;
T.Y      = DEC(:);
T.DayNum = dayGrid(:);
T.DayC   = T.DayNum - mean(1:nDay);
T.ChanID = categorical(chanGrid(:));
T.ROI    = ROI(chanGrid(:));

% ROI × Day interaction model (random intercept per channel)
lme_roi = fitlme(T, 'Y ~ 1 + DayC*ROI + (1|ChanID)', 'FitMethod','REML');

disp(lme_roi)
anova(lme_roi)   % look at DayC:ROI terms

% plotting
%%%% slops with sig. 
outTbl = plot_roi_slopes(T,lme_roi);

%%%% plotting at specific ROIs
m1_dec=[];
for i=1:10
    tmp=DEC(:,i);
    tmp = tmp(m1(:));
    m1_dec(i) = median(tmp);
end
X=[1:10];
Y = m1_dec;
figure;plot(X,Y,'.k','MarkerSize',20)
mdl=fitlm(X,Y);
bhat = mdl.Coefficients.Estimate;
yhat =  mdl.Fitted;
yhat1 = [ones(length(X),1) X(:)]*bhat;
hold on
plot(X,yhat,'k','LineWidth',1)
plot_beautify
xlabel('Days')
ylabel('high gamma decoding information')
title('M1')
xticks(1:10)
xlim([0.5 10.5])

% plotting brain regions
% plot cortex and only specific channels 
% m1 channels
ch_wts = zeros(253,1);
ch_wts(m1(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)
% pmd channels
ch_wts = zeros(253,1);
ch_wts(pmv(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)
% stg1
ch_wts = zeros(253,1);
ch_wts(tg1(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)
% stg2
ch_wts = zeros(253,1);
ch_wts(tg2(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)
% spchm1
ch_wts = zeros(253,1);
ch_wts(spchm1(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)
%spchpmd
ch_wts = zeros(253,1);
ch_wts(spch_pmv(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)
%ventral to m1/s1
ch_wts = zeros(253,1);
ch_wts(lm1(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)
%ventral to pmd
ch_wts = zeros(253,1);
ch_wts(lpmv1(:))=1;
plot_on_brain1(ch_wts,cortex,elecmatrix,ecog_grid_253)

%%%% MISC STUFF 
tmp=angle(mean(pac));
I=find(pac_tmp>0.3);
figure;rose(tmp(I))
title('Preferred angle between hG and alpha')
plot_beautify


% % plot mahab distances days
for i=1:size(mahab_dist_days,1)

    mahab_dist_day1 = mahab_dist_days(i,:);
    ch_wts1 = [mahab_dist_day1(1:107) 0 mahab_dist_day1(108:111) 0  mahab_dist_day1(112:115) 0 ...
        mahab_dist_day1(116:end)];

    plot_on_brain(ch_wts1,cortex,elecmatrix,ecog_grid)
    title(['hG decoding info CL Day ' num2str(i)])

end

%% (B1 ARROW 253 GRID): LOOKING at the the relationship b/w decoding 
% information at each channel and % PAC b/w hg and alpha/delta at that 
% channel





clc;clear
close all

if ispc
    root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker';
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
    cd(root_path)
    addpath('C:\Users\nikic\Documents\MATLAB\DrosteEffect-BrewerMap-5b84f95')    
    addpath 'C:\Users\nikic\Documents\MATLAB'
    load('F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B3\ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves\helpers'))
    imaging_B1_253;close all

else
    root_path ='/media/reza/ResearchDrive/ECoG_BCI_TravelingWave_HandControl_B3_Project/Data';
    cd(root_path)
    load session_data_B3_Hand
    load('ECOG_Grid_8596_000067_B3.mat')
    addpath(genpath('/home/reza/Repositories/ECoG_BCI_TravelingWaves/helpers'))

end


% d1 = designfilt('bandpassiir','FilterOrder',4, ...
%     'HalfPowerFrequency1',8,'HalfPowerFrequency2',10, ...
%     'SampleRate',1e3);

d1 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',1e3); % or 7 to 9 for b1 253 grid


d2 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',70,'HalfPowerFrequency2',150, ...
    'SampleRate',1e3);

d3 = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',4, ...
    'SampleRate',1e3);


folders={'20240515', '20240517', '20240614', '20240619', '20240621', '20240626'};



reg_days=[];
mahab_dist_days=[];
plot_true = true;
pac_days=[];

for i=1:length(folders)


    folderpath = fullfile(root_path,folders{i},'Robot3DArrow');
    D= dir(folderpath);
    D = D(3:end);
    imag_idx=[];
    online_idx=[];
    for j=1:length(D)
        subfoldername = dir(fullfile(folderpath,D(j).name));
        if strcmp(subfoldername(3).name,'Imagined')
            imag_idx=[imag_idx j];
        elseif strcmp(subfoldername(3).name,'BCI_Fixed')
            online_idx=[online_idx j];
        end
    end



    % %%%%% get imagined data files
    % folders = session_data(i).folders(imag_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandImagined',folders{ii},'Imagined');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' OL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')


    %%%%%% getting online files now     
    files=[];
    for ii=1:length(online_idx)
        imag_folderpath = fullfile(folderpath, D(online_idx(ii)).name,'BCI_Fixed');
        files = [files;findfiles('mat',imag_folderpath)'];
    end

    % get the phase locking value
    disp(['Processing Day ' num2str(i) ' CL'])
    [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %pac=zeros(100,253);
    pac_days(i,:)=abs(mean(pac));


    % get the mahab dist at each channel
    [mahab_dist] = get_mahab_dist_7DoF(files);
    mahab_dist_days(i,:) = mahab_dist;


    if plot_true

        figure;
        % plot pac as brain image
        pac_tmp = abs(mean(pac));
        ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
            pac_tmp(116:end)];
        %figure;
        subplot(1,3,1)
        imagesc(ch_wts(ecog_grid))
        title(['PAC Day ' num2str(i)])
        ch_wts_pac=ch_wts;        

        % plot mahab dist as brain image
        ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
            mahab_dist(116:end)];
        %figure;
        subplot(1,3,2)
        imagesc(ch_wts(ecog_grid))
        title(['Mahab dist Day ' num2str(i)])
        ch_wts_mahab=ch_wts;

        % hand knob weights
        %hnd2 = ch_wts(ecog_grid(2:3,3:4));


        % plot and see?
        %figure;
        subplot(1,3,3)
        plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
        xlabel('Mahab Dist')
        ylabel('PAC')
        title(['CL Day ' num2str(i)])
        ylim([0 0.7])
        plot_beautify
        hold on

    end




    % regression
    y = abs(mean(pac))';
    x = mahab_dist';
    x = [ones(size(x,1),1) x];
    [B,BINT,R,RINT,STATS1] = regress(y,x);
   

    if plot_true

        yhat = x*B;
        plot(x(:,2),yhat,'k')

        % plot mahab dist on brain
        ch_wts_mahab=mahab_dist;
        plot_on_brain_B1_253(ch_wts_mahab,cortex,elecmatrix,ecog_grid,15)
          title(['hG decoding info CL Day ' num2str(i)])

        % plot PAC on brain
        ch_wts_pac = abs(mean(pac));
        plot_on_brain_B1_253(ch_wts_pac,cortex,elecmatrix,ecog_grid,25)
        title(['hG-delta PAC CL Day ' num2str(i)])

    end

    reg_days(:,i) = [B; STATS1(3)];

    %
    % %%%%%%%%% getting batch files now
    % folders = session_data(i).folders(batch_idx);
    % day_date = session_data(i).Day;
    % files=[];
    % for ii=1:length(folders)
    %     folderpath = fullfile(root_path, day_date,'HandOnline',folders{ii},'BCI_Fixed');
    %     %cd(folderpath)
    %     files = [files;findfiles('mat',folderpath)'];
    % end
    %
    % % get the phase locking value
    % disp(['Processing Day ' num2str(i) ' CL'])
    % [pac,alpha_phase,hg_alpha_phase] = compute_pac(files,d1,d2);
    %
    %
    % % get the mahab dist at each channel
    % [mahab_dist] = get_mahab_dist(files);
    %
    %
    % % plot pac on brain
    % pac_tmp = abs(mean(pac));
    % ch_wts = [pac_tmp(1:107) 0 pac_tmp(108:111) 0  pac_tmp(112:115) 0 ...
    %     pac_tmp(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % plot mahab dist on brain grid
    % ch_wts = [mahab_dist(1:107) 0 mahab_dist(108:111) 0  mahab_dist(112:115) 0 ...
    %     mahab_dist(116:end)];
    % figure;
    % imagesc(ch_wts(ecog_grid))
    %
    % % hand knob weights
    % hnd9b = ch_wts(ecog_grid(2:3,3:4));
    %
    %
    % % plot and see?
    % figure;
    % plot(mahab_dist,abs(mean(pac)),'.','MarkerSize',20)
    % xlabel('Mahab Dist')
    % ylabel('PAC')
    % title(['CL Day ' num2str(i)])
    % ylim([0 0.7])
    % plot_beautify
    %


end

% plotting 

figure;hold on
plot(log(reg_days(3,:)),'.','MarkerSize',20)
xlabel('Days')
ylabel('Log P-value')
xticks(1:size(reg_days,2))
y = log(reg_days(3,:))';
x = (1:size(reg_days,2))';
x = [ones(size(x,1),1) x];
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
%plot(x(:,2),yhat,'k')
plot_beautify
title('Evolution of alpha-hG PAC and discriminability (sig)')
hline(log(0.05),'--r')



figure;hold on
plot((reg_days(2,:)),'.','MarkerSize',20)
xlabel('Days')
ylabel('Slope b/w PAC and decoding info.')
xticks(1:size(reg_days,2))
y = reg_days(2,:)';
x = (1:size(reg_days,2))';
x = [ones(size(x,1),1) x];
[B,BINT,R,RINT,STATS1] = regress(y,x);
yhat = x*B;
plot(x(:,2),yhat,'k')
plot_beautify
title('Evolution of delta-hG PAC and discriminability')
xlim([0.5 6.5])


% tmp=angle(mean(pac));
% I=find(pac_tmp>0.3);
% figure;rose(tmp(I))
% title('Preferred angle between hG and alpha')
% plot_beautify



save mahab_pac_delta_hg_B1_253_7DoF -v7.3


