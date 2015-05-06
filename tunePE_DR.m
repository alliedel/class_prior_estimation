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

[n1, ~] = size(X);
[n2, m2] = size(y);
if n1 ~= n2
    error('Number of training samples and corresponding labels must be equal')
elseif m2 ~= 1
    error('Labels must be a vector')
else

k = 4;  % number of folds

%% Gradient Descent
step_size = 1;
sigma = 1;
lambda = 0.01;
max_iters = 10;
dsigma = 0.01;
dlambda = 0.01;
iters = 0;
dMSE = Inf;

% Validate on multiple priors
validation_priors = [0.1 0.9];
X_validation = cell(size(validation_priors));
y_validation = cell(size(validation_priors));
partitions = cell(size(validation_priors));
classes = sort(unique(y));
indices_neg = find(y==classes(1));
indices_pos = find(y==classes(2));
for iPrior = 1:length(validation_priors)
    validation_prior = validation_priors(iPrior);
    train_size = floor(min(length(indices_neg)/(1-validation_prior), length(indices_pos)/validation_prior));
    indices_train = [datasample(indices_neg, round((1-validation_prior)*train_size));
                     datasample(indices_pos, round(validation_prior*train_size))];
    X_validation{iPrior} = X(indices_train, :);
    y_validation{iPrior} = y(indices_train);
    partitions{iPrior} = cvpartition(length(y_validation{iPrior}),'KFold',k);
end

% Compute initial MSE
MSE = 0;
for iPrior = 1:length(validation_priors)
    for iPartition = 1:k
        c = partitions{iPrior};
        X_train = X_validation{iPrior}(~c.test(iPartition), :);
        y_train = y_validation{iPrior}(~c.test(iPartition));
        X_test = X_validation{iPrior}(c.test(iPartition), :);
        y_test = y_validation{iPrior}(c.test(iPartition));

        [estimatedPriors, ~] = computePE_DR(X_train, y_train, X_test, sigma, lambda+dlambda);
        truePriors = NaN(size(estimatedPriors));
        for iClass = 1:length(classes)
            class = classes(iClass);
            py = sum(y_test == class)/length(y_test);
            truePriors(iClass,:) = [class, py];
        end
        MSE = MSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
    end
end
MSE = MSE/(length(validation_priors)*k);

while iters<max_iters %&& dMSE>0.001
    % Step along sigma
    dMSE = 0;
    for iPrior = 1:length(validation_priors)
        for iPartition = 1:k
            c = partitions{iPrior};
            X_train = X_validation{iPrior}(~c.test(iPartition), :);
            y_train = y_validation{iPrior}(~c.test(iPartition));
            X_test = X_validation{iPrior}(c.test(iPartition), :);
            y_test = y_validation{iPrior}(c.test(iPartition));
            
            [estimatedPriors, ~] = computePE_DR(X_train, y_train, X_test, sigma+dsigma, lambda);
            truePriors = NaN(size(estimatedPriors));
            for iClass = 1:length(classes)
                class = classes(iClass);
                py = sum(y_test == class)/length(y_test);
                truePriors(iClass,:) = [class, py];
            end
            dMSE = dMSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
        end
    end
    dMSE = dMSE/(length(validation_priors)*k);
    gradient = (dMSE-MSE)/dsigma;
    sigma = sigma - step_size*gradient;
    if sigma <= 0
        sigma = 0.0001;
    end
    
    % Step along lambda
    dMSE = 0;
    for iPrior = 1:length(validation_priors)
        for iPartition = 1:k
            c = partitions{iPrior};
            X_train = X_validation{iPrior}(~c.test(iPartition), :);
            y_train = y_validation{iPrior}(~c.test(iPartition));
            X_test = X_validation{iPrior}(c.test(iPartition), :);
            y_test = y_validation{iPrior}(c.test(iPartition));
            
            [estimatedPriors, ~] = computePE_DR(X_train, y_train, X_test, sigma, lambda+dlambda);
            truePriors = NaN(size(estimatedPriors));
            for iClass = 1:length(classes)
                class = classes(iClass);
                py = sum(y_test == class)/length(y_test);
                truePriors(iClass,:) = [class, py];
            end
            dMSE = dMSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
        end
    end
    dMSE = dMSE/(length(validation_priors)*k);
    gradient = (dMSE-MSE)/dlambda;
    lambda = lambda - step_size*gradient;
    if lambda <= 0
        lambda = 0.00001;
    end
    
    % Compute new MSE
    new_MSE = 0;
    for iPrior = 1:length(validation_priors)
        for iPartition = 1:k
            c = partitions{iPrior};
            X_train = X_validation{iPrior}(~c.test(iPartition), :);
            y_train = y_validation{iPrior}(~c.test(iPartition));
            X_test = X_validation{iPrior}(c.test(iPartition), :);
            y_test = y_validation{iPrior}(c.test(iPartition));
            
            [estimatedPriors, ~] = computePE_DR(X_train, y_train, X_test, sigma, lambda+dlambda);
            truePriors = NaN(size(estimatedPriors));
            for iClass = 1:length(classes)
                class = classes(iClass);
                py = sum(y_test == class)/length(y_test);
                truePriors(iClass,:) = [class, py];
            end
            new_MSE = new_MSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
        end
    end
    new_MSE = new_MSE/(length(validation_priors)*k);
    dMSE = abs(new_MSE - MSE);
    MSE = new_MSE;
    iters = iters+1;
end
            
%% Cross-validate over a range of parameters
% sigmas = [0.1 1.0 5.0 15.0];
% lambdas = [0.01, 0.1, 1.0];
% sigma = sigmas(1);
% lambda = lambdas(1);
% max_iters = 100;
% MSEs = NaN(length(sigmas), length(lambdas));
% classes = sort(unique(y));
% sigma_changed = true;
% lambda_changed = true;
% iters = 0;

% while sigma_changed && lambda_changed && iters<max_iters
%     if sigma_changed
%         % Update lambda
%         for iLambda = 1:length(lambdas)
%             new_lambda = lambdas(iLambda);
%             MSEs = NaN(size(lambdas));
%             MSE = 0;
%             for iPartition = 1:k
%                 X_train = X(~c.test(iPartition), :);
%                 y_train = y(~c.test(iPartition));
%                 X_test = X(c.test(iPartition), :);
%                 y_test = y(c.test(iPartition));
%                 [estimatedPriors, ~] = computePE_DR(X_train, y_train, X_test, sigma, new_lambda);
%                 truePriors = NaN(size(estimatedPriors));
%                 for iClass = 1:length(classes)
%                     class = classes(iClass);
%                     py = sum(y_test == class)/length(y_test);
%                     truePriors(iClass,:) = [class, py];
%                 end
%                 MSE = MSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
%             end
%             MSE = MSE/k;
%             MSEs(iLambda) = MSE;
%         end
%         [~, i] = min(MSEs);
%         new_lambda = lambdas(i);
%         sigma_changed = false;
%         if new_lambda == lambda
%             lambda_changed = false;
%         else
%             lambda_changed = true;
%             lambda = new_lambda;
%         end
%     elseif  lambda_changed
%         %Updated sigma
%         for iSigma = 1:length(sigmas)
%             new_sigma = sigmas(iSigma);
%             MSEs = NaN(size(sigmas));
%             MSE = 0;
%             for iPartition = 1:k
%                 X_train = X(~c.test(iPartition), :);
%                 y_train = y(~c.test(iPartition));
%                 X_test = X(c.test(iPartition), :);
%                 y_test = y(c.test(iPartition));
%                 [estimatedPriors, ~] = computePE_DR(X_train, y_train, X_test, new_sigma, lambda);
%                 truePriors = NaN(size(estimatedPriors));
%                 for iClass = 1:length(classes)
%                     class = classes(iClass);
%                     py = sum(y_test == class)/length(y_test);
%                     truePriors(iClass,:) = [class, py];
%                 end
%                 MSE = MSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
%             end
%             MSE = MSE/k;
%             MSEs(iSigma) = MSE;
%         end
%         [~, i] = min(MSEs);
%         new_sigma = sigmas(i);
%         lambda_changed = false;
%         if new_sigma == sigma
%             sigma_changed = false;
%         else
%             sigma_changed = true;
%             sigma = sigma_lambda;
%         end
%     end
%     iters = iters+1;
% end

% %% Validate over 10/90 and 90/10 splits
% sigmas = [0.1 1.0 5.0 15.0];
% lambdas = [0.01, 0.1, 1.0];
% validation_priors = [0.1 0.9];
% indices_neg = find(y==classes(1));
% indices_pos = find(y==classes(2));
% for iPrior = 1:length(validation_priors)
%     validation_prior = validation_priors(iPrior);
% 
%     train_size = floor(min(length(indices_neg)/(1-validation_prior), length(indices_pos)/validation_prior));
%     indices_train = [datasample(indices_neg, round(0.5*train_size));
%                      datasample(indices_pos, round(0.5*train_size))];
%     X_subsample = X(indices_train, :);
%     y_subsample = y(indices_train);
% 
%     %% Split samples into folds
%     k = 4;  % number of folds
%     c = cvpartition(length(y_subsample),'KFold',k);
% 
%     %% Cross-validate over a range of parameters
%     MSEs = zeros(length(sigmas), length(lambdas));
%     classes = sort(unique(y));
%     for iSigma = 1:length(sigmas)
%         sigma = sigmas(iSigma);
%         for iLambda = 1:length(lambdas)
%             lambda = lambdas(iLambda);
%             MSE = 0;
%             for iPartition = 1:k
%                 X_train = X_subsample(~c.test(iPartition), :);
%                 y_train = y_subsample(~c.test(iPartition));
%                 X_test = X_subsample(c.test(iPartition), :);
%                 y_test = y_subsample(c.test(iPartition));
%                 [estimatedPriors, ~] = computePE_DR(X_train, y_train, X_test, sigma, lambda);
%                 truePriors = NaN(size(estimatedPriors));
%                 for iClass = 1:length(classes)
%                     class = classes(iClass);
%                     py = sum(y_test == class)/length(y_test);
%                     truePriors(iClass,:) = [class, py];
%                 end
%                 MSE = MSE + computeMSE(estimatedPriors(:,2), truePriors(:,2));
%             end
%             MSE = MSE/k;
%             MSEs(iSigma, iLambda) = MSEs(iSigma, iLambda) + MSE;
%         end
%     end
% end
% %% Select and return the best parameters
% [~, argmin] = min(MSEs(:));
% [iSigma, iLambda] = ind2sub(size(MSEs), argmin);
% sigma = sigmas(iSigma);
% lambda = lambdas(iLambda);
end

