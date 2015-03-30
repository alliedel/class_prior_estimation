classdef PE_DR
    %PE_DR Class prior estimation using Pearson divergence with density 
    %      ratio estimation [duPlessis2013]. Tunes kernel bandwidtdh and
    %      regularization parameters with cross validation
    %   Detailed explanation goes here
    
    properties
        sigma  % tuned kernel bandwidth
        lambda  % tuned regularization parameter
        Xtrain  % training samples
        ytrain  % training labels
    end
    methods
        function obj = PE_DR(X, y)
            if nargin ~= 2
                error('Invalid number of inputs')
            elseif ~ismatrix(X) || ~ismatrix(y) 
                error('Inputs must be matrices')
            else
                [n1, ~] = size(X);
                [n2, m2] = size(y);
                if n1 ~= n2
                    error('Number of training samples and corresponding labels must be equal')
                elseif m2 ~= 1
                    error('Labels must be a vector')
                else
                    [obj.sigma, obj.lambda] = obj.tune(X1, y1, X2);
                end
            end
        end
        function [sigma, lambda] = tune(X, y)
            [sigma, lambda] = tunePE_DR(X, y);
        end
        function priors = estimateClassBalance(obj, X)
            priors = computePE_DR(obj.Xtrain, obj.ytrain, X, obj.sigma, obj.lambda);
        end
    end
    
end

