%% FIGURE 4
% GOAL HERE IS TO LOAD THE PAC AND MU STATE POWER IN ARROW TASKS
% COMPARE MU HG PAC, LFO HG PAC AND SHOW THESE OSCILLATORY DYNAMICS AT THE
% TRIAL LEVEL AFTER SHOWING THE CONFUSION MATRICES


%% init


clear
clc


if ispc
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_TravelingWaves'))
    addpath(genpath('C:\Users\nikic\Documents\GitHub\ECoG_BCI_HighDim'))
else

    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
    addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
    cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves')
end


%% ERPs of Mu and hG and LFO


%% Percent sig Mu-hG PAC channels 
% contrast with LFO-hG PAC


save PAC_B1_LFO_hG_rawValues_New -v7.3
%save PAC_B1_Mu_hG_rawValues_New -v7.3

%% Decoding relationship w/ Mahab Dist

