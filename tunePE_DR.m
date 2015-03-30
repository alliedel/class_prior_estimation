function [ sigma, lambda ] = tunePE_DR( X, y)
%PE_DR Tune kernel bandwidth and regularization constant with k-fold cross
% validation
%   Inputs:
%       X - n x m matrix of training samples, where n is the number of
%       samples and m is the number of features
%       y - n x 1 vector of training class labels
%       X2 - n' x m matrix of testing samples
%
%   Output:
%       sigma - Kernel bandwidth
%       lambda - regularization constant

%% Split samples into folds
k = 10;  % number of folds

%% Cross-validate over a range of parameters

%% Select and return the best parameters



    




end

