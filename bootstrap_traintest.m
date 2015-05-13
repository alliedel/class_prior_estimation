function [ ci, bootstats] = bootstrap_traintest(nboot, bootfun, X_train, y_train, X_test, y_test, varargin)
%BOOTSTRAPPED_MSE Bootstraps training and testing data for PE_DR MSE statistics
%   Inputs:
%       nboot - number of bootstraps
%       bootfun - function handle. Accepts inputs (X_train, y_train,
%       X_test, y_test). Vector output.
%           if bootfun is cell array, additional inputs follow
%           ex. {bootfun, a, b} calls bootfun(X_train, y_train, X_test,
%           y_test, a, b)
%       X_train - training samples [n x m] (samples n, features m)
%       y_train - training labels [n x 1]
%       X_test - testing samples
%       y_test - testing labels
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

    if nargin<6
        error('Too few inputs');
    end;
    if nboot<=0 || nboot~=round(nboot)
        error('Invalid nboot input')
    end;
    if size(y_train,2)~=1
        error('Invalid training labels input')
    end;
    if size(X_train,1)~=size(y_train,1)
        error('Number of training samples and labels do not match')
    end;
    if size(y_test,2)~=1
        error('Invalid testing labels input')
    end;
    if size(X_test,1)~=size(y_test,1)
        error('Number of testing samples and labels do not match')
    end;
    if size(X_train,2)~=size(X_test,2)
        error('Number of features do not match')
    end;

    if ~iscell(bootfun) % default syntax
        data = {};
    else % syntax with optional type, alpha, nbootstd, and stderrfun name/value pairs
        data = bootfun(2:end);
        bootfun = bootfun{1};
    end

    p = inputParser;
    defaultAlpha = 0.05;
    defaultType = 'norm';
    expectedTypes = {'norm','per','cper'};

    addOptional(p,'alpha',defaultAlpha,@isnumeric);
    addOptional(p,'ci_type',defaultType,...
                 @(x) any(validatestring(x,expectedTypes)));
    parse(p,varargin{:});
    alpha = p.Results.alpha;
    ci_type = p.Results.ci_type;

    stat = bootfun(X_train, y_train, X_test, y_test, data{:});
    stat = stat(:);  % turn into column vector
    if ~isvector(stat)
        error('Output from bootfun must be vector')
    end

    classes = sort(unique(y_train));
    train_indices_neg = find(y_train==classes(1));
    train_indices_pos = find(y_train==classes(2));
    test_indices_neg = find(y_test==classes(1));
    test_indices_pos = find(y_test==classes(2));

    bootstats = NaN(nboot, length(stat));

    for i = 1:nboot
        % Force training and testing data to have same prior as original
        indices_train = [datasample(train_indices_neg, length(train_indices_neg), 'Replace', true);
                        datasample(train_indices_pos, length(train_indices_pos), 'Replace', true)];
        indices_test = [datasample(test_indices_neg, length(test_indices_neg), 'Replace', true);
                        datasample(test_indices_pos, length(test_indices_pos), 'Replace', true)];
        X_train_bootstrap = X_train(indices_train, :);
        y_train_bootstrap = y_train(indices_train);
        X_test_bootstrap = X_test(indices_test, :);
        y_test_bootstrap = y_test(indices_test, :);

        bstat = bootfun(X_train_bootstrap, y_train_bootstrap, X_test_bootstrap, y_test_bootstrap, data{:});
        bstat = bstat(:);  % column vector
        bootstats(i, :) = bstat';
    end

    switch ci_type
        case 'norm'; ci = ci_norm(stat, bstat, alpha);
        case 'per'; ci = ci_per(stat, bstat, alpha);
        case 'cper'; ci = ci_cper(stat, bstat, alpha);
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
function ci = ci_per(~,bstat,alpha)
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
