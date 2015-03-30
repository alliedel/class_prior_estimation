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
c = cvpartition(length(y),'KFold',k);

%% Cross-validate over a range of parameters
sigmas = [0.001, 0.01, 0.1, 1, 10];
lambdas = [0.001, 0.01, 0.1, 1, 10];
MSEs = NaN(length(sigmas), length(lambdas));
classes = sort(unique(y));
for iSigma = 1:length(sigmas)
    sigma = sigmas(iSigma);
    for iLambda = 1:length(lambdas)
        lambda = lambdas(iLambda);
        MSE = 0;
        for iPartition = 1:k
            X_train = X(~c.test(iPartition), :);
            y_train = y(~c.test(iPartition));
            X_test = X(c.test(iPartition), :);
            y_test = y(c.test(iPartition));
            estimatedPriors = computePE_DR(X_train, y_train, X_test, sigma, lambda);
            truePriors = NaN(size(estimatedPriors));
            for iClass = 1:length(classes)
                class = classes(iClass);
                py = sum(y_test == class)/length(y_test);
                truePriors(iClass,:) = [class, py];
            end
            MSE = MSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
        end
        MSE = MSE/k;
        MSEs(iSigma, iLambda) = MSE;
    end
end

%% Select and return the best parameters
[~, argmin] = min(MSEs(:));
[iSigma, iLambda] = ind2sub(size(MSEs), argmin);
sigma = sigmas(iSigma);
lambda = lambdas(iLambda);
end

