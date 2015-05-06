function [ whitened ] = whiten( features )
%WHITEN Zero mean and standard deviation of 1
%   Inputs:
%       features - nxm feature matrix (n is number of samples, m is number
%       of features)
%
%   Outputs:
%       whitened - same format as above, but features with zero mean and
%       standard deviation of 1

feature_mean = mean(features);
whitened = bsxfun(@plus, features, -feature_mean);
whitened_std = std(whitened);
whitened_std(whitened_std<1E-9) = 1;
whitened = bsxfun(@rdivide, whitened, whitened_std);


end

