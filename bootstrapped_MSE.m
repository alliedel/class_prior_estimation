function [ ci, bootstat] = bootstrapped_MSE(k, X_train, y_train, X_test, y_test, test_prior, sigma, lambda )
%BOOTSTRAPPED_MSE Bootstraps training and testing data for PE_DR MSE statistics
%   Inputs:
%       k - number of bootstraps
%       X_train - training samples [n x m] (samples n, features m)
%       y_train - training labels [n x 1]
%       X_test - testing samples
%       y_test - testing labels
%       test_prior - desired prior of the testing data
%       sigma - kernel bandwidth parameter
%       lambda - regularization parameter
%
%   Outputs:
%       MSEs - MSE at each bootstrap
%       pos_priors - positive prior at each bootstrap

classes = sort(unique(y_test));
indices_neg = find(y_test==classes(1));
indices_pos = find(y_test==classes(2));
max_test_size = floor(min(length(indices_neg)/(1-test_prior), length(indices_pos)/test_prior));
n = min(max_test_size, 1000);  % number of bootstrap samples to take

MSEs = NaN(k, 1);
pos_priors = NaN(k, 1);

data = struct();
data.X_train = X_train;
data.y_train = y_train;
data.X_test = X_test;
data.y_test = y_test;
data.indices_neg = indices_neg;
data.indices_pos = indices_pos;
data.classes = classes;
junk_fake_bootstrap = 1:10;
[ci, bootstat] = bootci(k, {@MSE_wrapper, data, junk_fake_bootstrap, ...
    test_prior, sigma, lambda}, 'alpha', 0.05, 'type', 'norm');  
% can't use BCA type (default) b/c doing double bootstrap (train and test)

end

function results = MSE_wrapper(data, ~, test_prior, sigma, lambda)
    X_train = data.X_train;
    y_train = data.y_train;
    X_test = data.X_test;
    y_test = data.y_test;
    indices_neg = data.indices_neg;
    indices_pos = data.indices_pos;
    classes = data.classes;
    max_test_size = floor(min(length(indices_neg)/(1-test_prior), length(indices_pos)/test_prior));
    n = min(max_test_size, 1000);  % number of bootstrap samples to take

    indices_test = [datasample(indices_neg, floor((1-test_prior)*n), 'Replace', true);
                    datasample(indices_pos, floor((test_prior)*n), 'Replace', true)];
    X_test_bootstrap = X_test(indices_test, :);
    y_test_bootstrap = y_test(indices_test);
    indices_train = datasample(1:length(y_train), 100, 'Replace', true);
    X_train_bootstrap = X_train(indices_train, :);
    y_train_bootstrap = y_train(indices_train);
    [prior, alphas] = computePE_DR(X_train_bootstrap, y_train_bootstrap, X_test_bootstrap, sigma, lambda);
    true_priors = NaN(size(prior, 1), 1);
    for iClass = 1:length(classes)
        true_priors(iClass) = sum(y_test_bootstrap==classes(iClass))/length(y_test_bootstrap);
    end
    pos_prior = prior(2,2);
    MSE = computeMSE(prior(:,2), true_priors);
    results = [MSE, pos_prior];
end
