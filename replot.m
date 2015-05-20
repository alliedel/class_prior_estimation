clear, clc, close all
datasets = {'synthetic'}; %, 'australian_scale', 'diabetes', 'german_scale', 'ionosphere_scale'};  % 'adult', 
dataset_display_names = containers.Map;
dataset_display_names('synthetic') = 'Synthetic';
dataset_display_names('adult') = 'Adult';
dataset_display_names('australian_scale') = 'Australian';
dataset_display_names('diabetes') = 'Diabetes';
dataset_display_names('german_scale') = 'German';
dataset_display_names('ionosphere_scale') = 'Ionosphere';

to_plot = {'PE-DR','CC','ACC','Max','X','T50','PA','SPA','EM', 'SCC','MS','MM','Oracle'};  % ,

for iData = 1:length(datasets)
    fig_mse = figure; hold on
    fig_rae = figure; hold on
    fig_kld = figure; hold on
    dataset = datasets{iData};
    data = load(['temp/', dataset, '.mat']);
    methods = data.methods;
    results = data.results;
    test_pos_priors = data.test_pos_priors;
    
    col = hsv(length(to_plot));
    counter = 1;
    for iMethod = 1:length(methods)
        method = methods{iMethod};
        if strmatch(method,to_plot) ~= 0
            mse_lower_ci = max(arrayfun(@(x) x.MSEs_ci(1, iMethod), results), 0);
            mse_upper_ci = max(arrayfun(@(x) x.MSEs_ci(2, iMethod), results), 0);
            MSEs = arrayfun(@(x) mean(x.MSEs(:, iMethod)), results);
            rae_lower_ci = max(arrayfun(@(x) x.RAEs_ci(1, iMethod), results), 0);
            rae_upper_ci = max(arrayfun(@(x) x.RAEs_ci(2, iMethod), results), 0);
            RAEs = arrayfun(@(x) mean(x.RAEs(:, iMethod)), results);
            kld_lower_ci = max(arrayfun(@(x) x.KLDs_ci(1, iMethod), results), 0);
            kld_upper_ci = max(arrayfun(@(x) x.KLDs_ci(2, iMethod), results), 0);
            KLDs = arrayfun(@(x) mean(x.KLDs(:, iMethod)), results);
            
            if strcmp(method, 'PE-DR')
                line_style = '--';
                line_width = 2;
            else
                line_style = '-';
                line_width = 1.3;
            end
            figure(fig_mse)
            plot(test_pos_priors, MSEs, line_style, 'color', col(counter,:), 'LineWidth', line_width); hold on
            %errorbar(test_pos_priors, MSEs, min(MSEs, MSEs-mse_lower_ci), mse_upper_ci-MSEs, 'color',col(iMethod,:)); hold on
            figure(fig_rae)
            plot(test_pos_priors, RAEs, line_style, 'color', col(counter,:), 'LineWidth', line_width); hold on
            %errorbar(test_pos_priors, RAEs, min(RAEs, RAEs-rae_lower_ci), rae_upper_ci-RAEs, 'color',col(iMethod,:)); hold on
            figure(fig_kld)
            plot(test_pos_priors, KLDs, line_style, 'color', col(counter,:), 'LineWidth', line_width); hold on
            %errorbar(test_pos_priors, KLDs, min(KLDs, KLDs-kld_lower_ci), kld_upper_ci-KLDs, 'color',col(iMethod,:)); hold on
            counter = counter + 1;
        end
    end

    figure(fig_mse)
    legend(to_plot);
    xlabel('True class prior')
    ylabel('Mean squared error')
    title(dataset_display_names(dataset));
    saveas(fig_mse, ['temp/', dataset, '_mse.eps'], 'epsc')
    
    figure(fig_rae)
    ylim([0, 1.5])
    legend(to_plot);
    xlabel('True class prior')
    ylabel('Relative Absolute Error')
    title(dataset_display_names(dataset));
    saveas(fig_mse, ['temp/', dataset, '_rae.eps'], 'epsc')
    
    figure(fig_kld)
    legend(to_plot);
    xlabel('True class prior')
    ylabel('Binomial Kullbach-Leibler')
    title(dataset_display_names(dataset));
    saveas(fig_mse, ['temp/', dataset, '_kld.eps'], 'epsc')
end

%% Trafficking data
fig_mse = figure;
data = load('results/trafficking.mat');
MSEs = data.MSEs;
MSEs_ci = data.MSEs_ci;
MSEs_ci(:,1) = min(MSEs_ci(:,1), 0);
MSEs_ci(:,2) = max(MSEs_ci(:,2), 0);
test_pos_priors = data.test_pos_priors;
h = errorbar(test_pos_priors, MSEs, min(MSEs, MSEs-MSEs_ci(:,1)), MSEs_ci(:,2)-MSEs, 'r--'); hold on
set(get(h,'Children'),{'LineWidth'},{2; 1})
legend('PE-DR')
xlabel('True test set prior');
ylabel('Mean squared error');
title('Trafficking Dataset')
saveas(fig_mse, 'temp/trafficking.eps', 'epsc')
