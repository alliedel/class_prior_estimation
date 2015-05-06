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
%[labels, features] = libsvmread('data/adult');
%[labels, features] = libsvmread('data/australian_scale');
%[labels, features] = libsvmread('data/diabetes');
%[labels, features] = libsvmread('data/german_scale');
%[labels, features] = libsvmread('data/ionosphere_scale');
[labels, features] = libsvmread('data/statlog_heart_scale');

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
true_test_pos_priors = NaN(size(test_pos_priors));
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

disp('True class priors:')
disp(true_test_pos_priors)

disp('Estimated class priors:')
disp(estimated_test_pos_priors)

errorbar(true_test_pos_priors, MSEs, MSEs-MSEs_ci(:,1), MSEs_ci(:,2)-MSEs, 'r-d')
ylim([0 0.3])
xlabel('True class prior')
ylabel('Mean squared error')



