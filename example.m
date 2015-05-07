clear, clc, close all
ttotal = tic;

% TODO: 10-fold cross validation

%% Allie's Synthetic Data
%[X_train, y_train, X_test, y_test] = getSyntheticData();


%% Matt's Synthetic Data
% dev = 0.2;
% dim = 5;
% features = [0.25 + dev*randn(200,dim);
%            rand(100,dim);
%            0.75 + dev*randn(200,dim);
%            rand(100,dim)];
% labels = [ones(300,1);
%            zeros(300,1)];


%% LIBSVM real world datasets
datasets = {'adult', 'australian_scale', 'diabetes', 'german_scale', 'ionosphere_scale'};
%dataset = 'adult'
%dataset = 'australian_scale';
%dataset = 'diabetes';
%dataset = 'german_scale';
%dataset = 'ionosphere_scale';
for iData = 1:length(datasets)
    dataset = datasets{iData};
    [labels, features] = libsvmread(['data/', dataset]);

    features(:,all(isnan(features))) = [];
    if any(any(isnan(features)))
        error('Erroneous NaN entry in feature file')
    end
    features = full(features);
    features = whiten(features);

    %% Tune parameters
    %[sigma, lambda] = tunePE_DR(X_train, y_train);
    sigma = 5;
    lambda = 0.001;

    %% Remove evenly balanced data for training
    classes = sort(unique(labels));
    train_size = 100;
    indices_neg = find(labels==classes(1));
    indices_pos = find(labels==classes(2));
    indices_train = [datasample(indices_neg, round(0.5*train_size));
                     datasample(indices_pos, round(0.5*train_size))];
    X_train = features(indices_train, :);
    y_train = labels(indices_train);
    features(indices_train, :) = [];
    labels(indices_train) = [];
    X_test = features;
    y_test = labels;



    %% Direct density ratio method
    test_pos_priors = [0.1:0.1:0.9];
    k = 20;  % number of bootstraps
    estimated_test_pos_priors = NaN(size(test_pos_priors));
    MSEs = NaN(size(test_pos_priors))';
    MSEs_ci = NaN(length(MSEs), 2);
    classes = sort(unique(y_test));
    for iPrior = 1:length(test_pos_priors)
        test_prior = test_pos_priors(iPrior);
        [ci, bootstat] = bootstrapped_MSE(k, X_train, y_train, X_test, y_test, test_prior, sigma, lambda);
        estimated_test_pos_priors(iPrior) = mean(bootstat(:,2));
        MSEs(iPrior) = mean(bootstat(:,1));
        MSEs_ci(iPrior, :) = ci(:,1)';
    end


    % scatter(X_train, 0.5*ones(size(X_train)), 50*ones(size(X_train)), y_train+1, 'x', 'LineWidth',1.3); hold on
    % scatter(X_test, 1.5*ones(size(X_test)), 50*ones(size(X_test)), y_test+1, 'x', 'LineWidth',1.3); hold on
    % axis([-0.2 1.2 0.45 1.55])
    % 
    % x = linspace(0,1,10000);
    % density_ratios = NaN(size(x));
    % for i = 1:length(x)
    %     phi_bold = evaluateBasis(x(i), X_train, estimator.sigma);
    %     density_ratios(i) = phi_bold*alphas;
    % end
    % plot(x, density_ratios, 'g')
    disp('Total run time is:')
    toc(ttotal)

    disp('Estimated class priors:')
    disp(estimated_test_pos_priors)

    errorbar(test_pos_priors, MSEs, MSEs-MSEs_ci(:,1), MSEs_ci(:,2)-MSEs)
    ylim([0 0.3])
    xlabel('True class prior')
    ylabel('Mean squared error')
    title(dataset);
    save(dataset, 'MSEs', 'MSEs_ci', 'test_pos_priors', 'estimated_test_pos_priors');

end



