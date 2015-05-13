%% Class prior estimation examples
%   Multiple datasets and all methods w/ bootstrapping

clear, clc, close all
ttotal = tic;

%% Load data 
datasets = {'synthetic'}; %, 'australian_scale', 'diabetes', 'german_scale', 'ionosphere_scale', 'trafficking'};  %  
%  'adult': NaN for logistic regression coefficients in competitors
%  (perhaps b/c too few training samples)

for iData = 1:length(datasets)
    dataset = datasets{iData};
    if strcmp(dataset, 'trafficking')
        features = csvread('data/trafficking/features_subsample.csv');
        labels = csvread('data/trafficking/labels_subsample.csv');
        %disp('Running PCA')
        %[coeff,score] = princomp(features);
        %features = score(:,1:300);
    elseif strcmp(dataset, 'synthetic')
        dev = 0.2;
        dim = 5;
        features = [0.25 + dev*randn(200,dim);
                   rand(100,dim);
                   0.75 + dev*randn(200,dim);
                   rand(100,dim)];
        labels = [ones(300,1);
                   zeros(300,1)];
    else
        [labels, features] = libsvmread(['data/', dataset]);
    end

    % Clean up NaN and 0 entries if present (found in libsvm datasets)
    features(:,all(isnan(features))) = [];
    features(:,all(features==0)) = [];
    if any(any(isnan(features)))
        error('Erroneous NaN entry in feature file')
    end
    features = full(features);
    features = whiten(features);


    %% Hyper-parameters
    %[sigma, lambda] = tunePE_DR(X_train, y_train);
    lambda = 0.001;
    sigma = 10;

    %% Sample training data
    train_size = round(0.8*size(features, 1));
    classes = sort(unique(labels));
    indices_neg = find(labels==classes(1));
    indices_pos = find(labels==classes(2));
    indices_train = [datasample(indices_neg, round(0.5*train_size));
                     datasample(indices_pos, round(0.5*train_size))];
    X_train = features(indices_train, :);
    y_train = labels(indices_train);
    features(indices_train, :) = [];
    labels(indices_train) = [];

    %% Run class prior estimation over range of priors
    test_pos_priors = 0.1:0.1:0.9;
    k = 20;  % number of bootstraps
    results = struct('MSEs', {}, 'MSEs_ci', {}, 'priors', {}, 'priors_ci', {});
    for iPrior = 1:length(test_pos_priors)
        % Sample testing data
        test_prior = test_pos_priors(iPrior);
        indices_neg = find(labels==classes(1));
        indices_pos = find(labels==classes(2));
        maxn = floor(min(length(indices_neg)/(1-test_prior), length(indices_pos)/test_prior));
        indices_test = [datasample(indices_neg, floor(maxn*(1-test_prior)));
                         datasample(indices_pos, floor(maxn*test_prior))];
        X_test = features(indices_test, :);
        y_test = labels(indices_test);
        
        % Run method
        [ci, bootstat] = bootstrap_traintest(k, {@all_methods, sigma, ...
            lambda}, X_train, y_train, X_test, y_test, 'alpha', 0.05, ...
            'ci_type', 'norm');
        
        results(iPrior).priors = bootstat(:, 1:size(bootstat, 2)/2);
        results(iPrior).MSEs = bootstat(:, size(bootstat, 2)/2+1:end);
        results(iPrior).priors_ci = ci(:, 1:size(bootstat, 2)/2);
        results(iPrior).MSEs_ci = ci(:, size(bootstat, 2)/2+1:end);
    end

    %% Results
    % Calculate MSEs
    disp('Total run time is:')
    toc(ttotal)

    disp('Estimated class priors:')
    disp(cell2mat(arrayfun(@(x) mean(x.priors), results, 'UniformOutput', false)'));

    output_path = ['results/', dataset];
    methods = {'PE-DR', 'CC', 'ACC','Max','X','T50','MS','MM','PA','SPA','SCC','EM'};
    save(output_path, 'results', 'test_pos_priors', 'methods');
end
%% Plot results
% figure, hold on
% col = [51,34,136
% 102,153,204
% 136,204,238
% 68,170,153
% 17,119,51
% 153,153,51
% 221,204,119
% 102,17,0
% 204,102,119
% 170,68,102
% 136,34,85
% 170,68,153]/255;
% for iMethod = 1:size(results(1).MSEs_ci, 2)
%     lower_ci = max(arrayfun(@(x) x.MSEs_ci(1, iMethod), results), 0);
%     upper_ci = max(arrayfun(@(x) x.MSEs_ci(2, iMethod), results), 0);
%     MSEs = arrayfun(@(x) mean(x.MSEs(:, iMethod)), results);
%     plot(test_pos_priors, MSEs, 'color', col(iMethod,:), 'LineWidth', 1.3); hold on
%     %errorbar(test_pos_priors, MSEs, min(MSEs, MSEs-lower_ci), upper_ci-MSEs, 'color',col(iMethod,:)); hold on
% end
% legend(methods);
% xlabel('True class prior')
% ylabel('Mean squared error')
% title(dataset);







