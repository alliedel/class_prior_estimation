function [ priors, alphas ] = computePE_DR( X1, y1, X2, sigma, lambda )
%PE_DR Class prior estimation using Pearson divergence with density ratio
%estimation [duPlessis2013]
%   Inputs:
%       X1 - n x m matrix of samples, where n is the number of
%       samples and m is the number of features
%       y1 - n x 1 vector of training class labels
%       X2 - n' x m matrix of testing samples
%       sigma - Kernel bandwidth
%       lambda - regularization constant
%   Output:
%       priors - c x 2 vector of estimated class priors, where c is the 
%       number of classes, column 1 is the class label and column 2 is the
%       estimated prior
    disp('Computing PE_DR...')
    classes = sort(unique(y1));
    R = eye(size(X1, 1)+1);
    R(1, 1) = 0;  % do not regularize constant basis function
    G = computeG(X1, X2, sigma);
    H = computeH(X1, y1, sigma, classes);
    tic
    opts.SYM = true;
    GR_inv_G = linsolve(G+lambda*R, G, opts);  toc %(G+lambda*R)\G; toc
    GR_inv_H = linsolve(G+lambda*R, H, opts);  toc %(G+lambda*R)\H; toc
    %% CVX Optimization
    c = length(classes);
    Q1 = H'*GR_inv_G*GR_inv_H;
    Q2 = H'*GR_inv_H;
    Q = -0.5*Q1 + Q2;
    cvx_begin
        variable theta(c)
        minimize( theta'*Q*theta - 0.5 )
        subject to
            ones(size(classes))' * theta == 1
    cvx_end
    priors = [classes, theta];
    alphas = GR_inv_H*theta;
end

function G_hat = computeG(X_train, X_test, sigma)
% Compute G_hat
    n_prime = size(X_test, 1);
    phi_bold_matrix = evaluateBasis(X_test, X_train, sigma);
    G_hat = phi_bold_matrix'*phi_bold_matrix/n_prime;
end

function H_hat = computeH(X_train, y_train, sigma, classes)
% Compute H_hat
    [n1, ~] = size(X_train);
    [n2, m2] = size(y_train);
    if m2 ~= 1
        error('Classes must be row vector')
    elseif n1 ~= n2
        error('Number of labels and samples must be equal')
    end
    H_hat = zeros(length(y_train)+1, length(classes));
    phi_bold_matrix = evaluateBasis(X_train, X_train, sigma);
    for i = 1:length(classes)
        class = classes(i);
        indices = y_train==class;
        h_y = sum(phi_bold_matrix(indices,:), 1)/sum(indices);
        H_hat(:, i) = h_y';
    end
end


