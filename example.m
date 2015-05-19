%% Class prior estimation examples
%   Multiple datasets and all methods w/ bootstrapping

clear, clc, close all
ttotal = tic;

%% Load data 
datasets = {'diabetes'}; %, 'australian_scale', 'diabetes', 'german_scale', 'ionosphere_scale', 'trafficking'};  %  
%  'adult': NaN for logistic regression coefficients in competitors
%  (perhaps b/c too few training samples)

for iData = 1:length(datasets)
    %% Load and clean dataset
    dataset = datasets{iData};
    if strcmp(dataset, 'trafficking')
        features = csvread('data/trafficking/features_subsample.csv');
        labels = csvread('data/trafficking/labels_subsample.csv');
        %disp('Running PCA')
        %[coeff,score] = princomp(features);
        %features = score(:,1:300);
    elseif strcmp(dataset, 'synthetic')
        dev = 0.2;
        dim = 1;
        ind = @(x) all(0 <= x, 2) & all(x <= 1, 2);
        pos_pdf = @(x) 1/3*ind(x) + 2/3*mvnpdf(x,0.25*ones(1,dim),0.02*eye(dim));
        neg_pdf = @(x) 1/3*ind(x) + 2/3*mvnpdf(x,0.75*ones(1,dim),0.02*eye(dim));
        features = [mvnrnd(0.25*ones(dim,1),0.02*eye(dim), 2000);
                    rand(1000, dim);
                    mvnrnd(0.75*ones(dim,1),0.02*eye(dim), 2000);
                    rand(1000, dim)];
        labels = [ones(3000,1);
                  zeros(3000,1)];
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
    if ~strcmp(dataset, 'synthetic')  % don't whiten synthetic data, need original PDFs for oracle 
        features = whiten(features);
    end


    %% Hyper-parameters
    %[sigma, lambda] = tunePE_DR(X_train, y_train);
    lambda = 0.001;
    sigma = 10;

    %% Run class prior estimation over range of priors
    test_pos_priors = [0.001, 0.01, 0.1:0.1:0.9, 0.99, 0.999];
    k = 20;  % number of bootstraps
    train_size = min(500, floor(0.5*length(labels)));
    train_prior = 0.5;
    results = struct('MSEs', {}, 'MSEs_ci', {}, 'priors', {}, 'priors_ci', {});
    for iPrior = 1:length(test_pos_priors)      
        test_prior = test_pos_priors(iPrior);
        if ~strcmp(dataset, 'synthetic')
            [ci, bootstat] = bootstrap_traintest(k, {@all_methods, sigma, ...
                lambda}, features, labels, train_size, train_prior, test_prior, 'alpha', 0.05, ...
                'ci_type', 'per');
        else  % if synthetic data, also run with Oracle model
            [ci, bootstat] = bootstrap_traintest(k, {@all_methods, sigma, ...
                lambda, pos_pdf, neg_pdf}, features, labels, train_size, train_prior, test_prior, 'alpha', 0.05, ...
                'ci_type', 'per'); 
        end
        
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

    output_path = ['temp/', dataset];
    if strcmp(dataset, 'synthetic')
        methods = {'PE-DR', 'CC', 'ACC','Max','X','T50','MS','MM','PA','SPA','SCC','EM','Oracle'};
    else
        methods = {'PE-DR', 'CC', 'ACC','Max','X','T50','MS','MM','PA','SPA','SCC','EM'};
    end
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







