function [ ci, bootstats] = bootstrap_traintest(nboot, bootfun, X, y, train_size, train_prior, test_prior, varargin)
%BOOTSTRAPPED_TRAINTEST Simultaneously bootstraps training and testing data
%   Inputs:
%       nboot - number of bootstraps
%       bootfun - function handle. Accepts inputs (X_train, y_train,
%       X_test, y_test). Vector output.
%           if bootfun is cell array, additional inputs follow
%           ex. {bootfun, a, b} calls bootfun(X_train, y_train, X_test,
%           y_test, a, b)
%       X - Samples [n x m] (samples n, features m)
%       y - Labels [n x 1]
%       train_size - int, number of training samples to use
%       train_prior - [0, 1]
%       test_prior - [0, 1] (uses as many test samples as possible, up to a
%       limit)
%       varargin - 'alpha' and 'ci_type'
%               alpha: confidence inteval parameter, default 0.05 (95%)
%               ci_type: confidence interval type, default 'norm'
%                        also available: 'per' and 'cper' (see below)
%
%   Outputs:
%       ci - Confidence intervals for each bootfun output (first row is
%       lower, second row is upper)
%       bootstats - Statistics for each bootstrap
%
%   Authors:
%       Matt Barnes
%       Mathworks

    if nargin<7
        error('Too few inputs');
    end;
    if nboot<=0 || nboot~=round(nboot)
        error('Invalid nboot input')
    end;
    if size(y,2)~=1
        error('Invalid labels input')
    end;
    if size(X,1)~=size(y,1)
        error('Number of samples and labels do not match')
    end;

    if ~iscell(bootfun) % default syntax
        data = {};
    else % syntax with optional type and alpha name/value pairs
        data = bootfun(2:end);
        bootfun = bootfun{1};
    end

    p = inputParser;
    defaultAlpha = 0.05;
    defaultType = 'per';
    expectedTypes = {'norm','per','cper'};

    addOptional(p,'alpha',defaultAlpha,@isnumeric);
    addOptional(p,'ci_type',defaultType,...
                 @(x) any(validatestring(x,expectedTypes)));
    parse(p,varargin{:});
    alpha = p.Results.alpha;
    ci_type = p.Results.ci_type;
    if ~strcmp(ci_type, 'per')
        error('Can only use percentile confidence intervals if resplitting training and testing data every bootstrap')
    end
    
    classes = sort(unique(y));
    indices_neg = find(y==classes(1));
    indices_pos = find(y==classes(2));
    
    %% Bootstrapping
    for i = 1:nboot
        % Split train and test data
        indices_train = [datasample(indices_neg, round(train_size*(1-train_prior)), 'Replace', true);
                        datasample(indices_pos, round(train_size*train_prior), 'Replace', true)];
        X_train_bootstrap = X(indices_train, :);
        y_train_bootstrap = y(indices_train);
        X_test_bootstrap = X;
        y_test_bootstrap = y;
        X_test_bootstrap(indices_train, :) = [];
        y_test_bootstrap(indices_train) = [];
        test_indices_neg = find(y_test_bootstrap==classes(1));
        test_indices_pos = find(y_test_bootstrap==classes(2));
        test_size = floor(min(length(test_indices_neg)/(1-test_prior), length(test_indices_pos)/test_prior));
        test_size = min(test_size, 1000);
        indices_test = [datasample(test_indices_neg, round(test_size*(1-test_prior)), 'Replace', true);
                        datasample(test_indices_pos, round(test_size*test_prior), 'Replace', true)];
        X_test_bootstrap = X_test_bootstrap(indices_test, :);
        y_test_bootstrap = y_test_bootstrap(indices_test);
        bstat = bootfun(X_train_bootstrap, y_train_bootstrap, X_test_bootstrap, y_test_bootstrap, data{:});
        bstat = bstat(:);  % column vector
        if ~exist('bootstats','var')
        	bootstats = NaN(nboot, length(bstat));
        end
        bootstats(i, :) = bstat';
    end

    switch ci_type
        case 'norm'; ci = ci_norm(stat, bootstats, alpha);
        case 'per'; ci = ci_per(bootstats, alpha);
        case 'cper'; ci = ci_cper(stat, bootstats, alpha);
    end
    
    % Reshape
    ci = reshape(ci,[2 size(bootstats, 2)]);

end   % bootstrap_traintest


function ci = ci_norm(stat,bstat,alpha)
% normal approximation interval
% A.C. Davison and D.V. Hinkley (1996), p198-200
% modified from Mathworks

    se = std(bstat,0,1);   % standard deviation estimate
    bias = mean(bsxfun(@minus,bstat,stat),1);
    za = norminv(alpha/2);   % normal confidence point
    lower = stat - bias + se*za; % lower bound
    upper = stat - bias - se*za;  % upper bound

    % return
    ci = [lower;upper];        
end   % bootnorm() 
 
%-------------------------------------------------------------------------
function ci = ci_per(bstat,alpha)
% percentile bootstrap CI
% modified from Mathworks

    pct1 = 100*alpha/2;
    pct2 = 100-pct1;
    lower = prctile(bstat,pct1,1); 
    upper = prctile(bstat,pct2,1);

    % return
    ci =[lower;upper];
end % bootper() 

%-------------------------------------------------------------------------
function ci = ci_cper(stat,bstat,alpha)
% corrected percentile bootstrap CI
% B. Efron (1982), "The jackknife, the bootstrap and other resampling
% plans", SIAM.
% modified from Mathworks

    % stat is transformed to a normal random variable z0.
    % z0 = invnormCDF[ECDF(stat)]
    z_0 = fz0(bstat,stat);
    z_alpha = norminv(alpha/2); % normal confidence point

    % transform z0 back using the invECDF[normCDF(2z0-za)] and
    % invECDF[normCDF(2z0+za)] 
    pct1 = 100*normcdf(2*z_0-z_alpha); 
    pct2 = 100*normcdf(2*z_0+z_alpha);

    % inverse ECDF
    m = numel(stat);
    lower = zeros(1,m);
    upper = zeros(1,m);
    for i=1:m
        lower(i) = prctile(bstat(:,i),pct2(i),1);
        upper(i) = prctile(bstat(:,i),pct1(i),1);
    end

    % return
    ci = [lower;upper];
end % bootcper()

function z0=fz0(bstat,stat)
% Compute bias-correction constant z0
% modified from Mathworks
    z0 = norminv(mean(bsxfun(@lt,bstat,stat),1) + mean(bsxfun(@eq,bstat,stat),1)/2);
end   % fz0()
