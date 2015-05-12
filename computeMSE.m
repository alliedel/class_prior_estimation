function [ MSE ] = computeMSE( priors1, priors2 )
%COMPUTEMSE Computes the mean squared error
%   Inputs:
%       priors1 - vector of priors, one per method
%       priors2 - vector of priors, classes in same order as priors1
%
%   Outputs:
%       MSE - the mean squared error

MSE = (priors1-priors2).^2;

end

