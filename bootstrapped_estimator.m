function [ ci, bootstat] = bootstrapped_estimator(k, X_train, y_train, X_test, y_test, test_prior, sigma, lambda )
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
%       ci - Prior confidence intervals for each method
%       priors - positive class prior estimates for each bootstrap and
%       method

classes = sort(unique(y_test));
indices_neg = find(y_test==classes(1));
indices_pos = find(y_test==classes(2));

data = struct();
data.X_train = X_train;
data.y_train = y_train;
data.X_test = X_test;
data.y_test = y_test;
data.indices_neg = indices_neg;
data.indices_pos = indices_pos;
junk_fake_bootstrap = 1:10;
[ci, bootstat] = bootci(k, {@prior_wrapper, data, junk_fake_bootstrap, ...
    test_prior, sigma, lambda}, 'alpha', 0.05, 'type', 'norm');  
% can't use BCA type (default) b/c doing double bootstrap (train and test)

end

function priors_MSE = prior_wrapper(data, ~, test_prior, sigma, lambda)
    X_train = data.X_train;
    y_train = data.y_train;
    X_test = data.X_test;
    indices_neg = data.indices_neg;
    indices_pos = data.indices_pos;
    max_test_size = floor(min(length(indices_neg)/(1-test_prior), length(indices_pos)/test_prior));
    n = min(max_test_size, 1000);  % number of bootstrap samples to take

    indices_test = [datasample(indices_neg, floor((1-test_prior)*n), 'Replace', true);
                    datasample(indices_pos, floor((test_prior)*n), 'Replace', true)];
    X_test_bootstrap = X_test(indices_test, :);
    indices_train = datasample(1:length(y_train), length(y_train), 'Replace', true);
    X_train_bootstrap = X_train(indices_train, :);
    y_train_bootstrap = y_train(indices_train);
    n = length(y_train_bootstrap);
    n = min(500, n);
    [prior, ~] = computePE_DR(X_train_bootstrap(1:n, :), y_train_bootstrap(1:n), X_test_bootstrap, sigma, lambda);
    competitors_priors = competitors(X_train_bootstrap, y_train_bootstrap, X_test_bootstrap);
    priors = [prior(2,2), competitors_priors];
    true_prior = floor((test_prior)*n)/length(indices_test);
    MSEs = computeMSE(priors, true_prior);
    if any(MSEs<0)
        print 'Invalid MSE'
    end
    priors_MSE = [priors, MSEs];
end
