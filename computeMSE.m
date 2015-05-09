function [ MSE ] = computeMSE( priors1, priors2 )
%COMPUTEMSE Computes the mean squared error
%   Inputs:
%       priors1 - vector of priors, one per class
%       priors2 - vector of priors, classes in same order as priors1
%
%   Outputs:
%       MSE - the mean squared error

MSE = sum((priors1(2)-priors2(2)).^2);

end

