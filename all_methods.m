function priors_MSEs = all_methods(X_train, y_train, X_test, y_test, varargin)  % sigma, lambda, pos_pdf, neg_pdf)
%ALL_METHODS Runs all the methods in succession
    
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
    MSEs = computeMSE(priors, true_prior);
    %per_error = computePerE(priors, true_prior);
    %kl_error = computeKLE(priors, true_prior);
    %info_gain = 
    priors_MSEs = [priors, MSEs];
end