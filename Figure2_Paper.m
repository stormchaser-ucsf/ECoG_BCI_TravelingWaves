%% init


clear
clc
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves/'))
addpath(genpath('/home/user/Documents/Repositories/ECoG_BCI_HighDim/'))
cd('/home/user/Documents/Repositories/ECoG_BCI_TravelingWaves')


%% LOAD IMAGING DATA FOR THE NEURON DATASET

imaging_EC189

imaging_EC210

imaging_EC176

%% Get hG and LFO ERPs example


%% PAC between hG and mu


%% PAC between hG and LFO 

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC176_ProcessingForNikhilesh/ecog_data_NN')
ec176 = load('EC176_sig_ch_LFO_hG_PAC.mat');

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC210')
ec210 = load('sig_ch_LFO_hG_PAC.mat');

cd('/media/user/Data/ecog_data/ECoG LeapMotion/Raw Data/EC189_ProcessingForNikhilesh/EC189')
ec189 = load('EC189_sig_ch_LFO_hG_PAC.mat');







