function priors_MSE_RAE_KLD = all_methods(X_train, y_train, X_test, y_test, varargin)  % sigma, lambda, pos_pdf, neg_pdf)
%ALL_METHODS Runs all the methods in succession, returns estimated priors
%and all metrics
    
    if length(varargin)==2
        sigma = varargin{1};
        lambda = varargin{2};
    elseif length(varargin)==4
        sigma = varargin{1};
        lambda = varargin{2};
        pos_pdf = varargin{3};
        neg_pdf = varargin{4};
    elseif ~isempty(varargin)
        error('Incorrect number of variable arguments')
    end
        
    classes = sort(unique(y_test));
    if length(classes) == 2
        true_prior = sum(y_test==classes(2))/length(y_test);
    elseif classes(1) == 1
        true_prior = 1;
    else
        true_prior = 0;
    end
    [prior1, ~] = computePE_DR(X_train, y_train, X_test, sigma, lambda);
    prior2 = classification_methods(X_train, y_train, X_test);
    if exist('pos_pdf','var')
        prior3 = oracle(X_test, pos_pdf, neg_pdf);
        priors = [prior1(2,2), prior2(:)', prior3];
    else
        priors = [prior1(2,2), prior2(:)'];
    end
    priors = min(priors, 1);
    priors = max(priors, 0);  % impose constraint priors must be [0,1]
    MSEs = computeMSE(true_prior, priors);
    RAEs = computeRAE(true_prior, priors);
    KLDs = computeKLD(true_prior, priors);
    priors_MSE_RAE_KLD = [priors, MSEs(:)', RAEs(:)', KLDs(:)'];
end

function relative_absolute_errors = computeRAE(true_prior, priors)
%COMPUTEPERE Computes the relative absolute error metric [Sebastiani2014]
% Inputs:
%   true_prior: the true prior
%   priors: vector of estimated priors
    true_neg_prior = 1-true_prior;
    neg_priors = 1-priors;
    relative_absolute_errors = 0.5*(abs(priors-true_prior)/true_prior + ...
        abs(neg_priors-true_neg_prior)/true_neg_prior);
end

function KLDs = computeKLD(true_prior, priors)
%COMPUTEKLE Computes the KL error, defined as the KL divergence between the
%binomial distributions [Sebastian2014]
    true_neg_prior = 1-true_prior;
    neg_priors = 1-priors;
    KLDs = 0.5*(true_prior*log(true_prior./priors) + ...
        true_neg_prior*log(true_neg_prior./neg_priors));
    if any(~isreal(KLDs))
        disp('Invalid KLD')
    end
end

function [ MSE ] = computeMSE(true_prior, priors)
%COMPUTEMSE Computes the mean squared error
%   Inputs:
%       true_prior - scalar of the true prior
%       priors - vector of estimated priors
%
%   Outputs:
%       MSE - the mean squared error

MSE = (true_prior-priors).^2;
end