clear, clc, close all
datasets = {'australian_scale', 'diabetes', 'german_scale', 'ionosphere_scale'};  % 'adult', 
dataset_names = {'Australian', 'Diabetes', 'German', 'Ionosphere'};
paper_MSEs = cell(size(datasets));
% paper_MSEs{1} = [0.10019011406844108, 0.04565217391304344
% 0.2, 0.035217391304347756
% 0.29980988593155894, 0.028260869565217395
% 0.4002851711026616, 0.025217391304347803
% 0.4994296577946768, 0.02434782608695646
% 0.5999049429657795, 0.028260869565217395
% 0.6997148288973383, 0.03565217391304348
% 0.8001901140684411, 0.04608695652173911
% 0.8993346007604561, 0.057826086956521694];  % adult
paper_MSEs{1} = [0.10069876228495264, 0.021062992125984226
0.19996053287094248, 0.02135826771653543
0.2998770695980176, 0.021062992125984226
0.39979878234201827, 0.021948818897637834
0.499714025064862, 0.02135826771653543
0.5996305617919372, 0.021062992125984226
0.6995483925232437, 0.021062992125984226
0.799468811263013, 0.02165354330708663
0.8993866419943195, 0.02165354330708663];  % australian_scale
paper_MSEs{2} = [0.1001393218653619, 0.07713791895234401
0.2000339302753615, 0.06839360227346702
0.2999295081218, 0.060386749742753265
0.4004872110560071, 0.0560679104074544
0.5003924832668347, 0.05543569935837167
0.60029969435054, 0.0562784166056155
0.700210783180001, 0.060070990445511674
0.8001286580645341, 0.06902581332254965
0.9000445940761895, 0.07650570790326139];  % diabetes
paper_MSEs{3} = [0.10019353776045237, 0.12695390781563126
0.1996733192412222, 0.09088176352705407
0.3004625689735636, 0.06803607214428858
0.39982430614653963, 0.057815631262525036
0.499843522661762, 0.053607214428857713
0.5991283937738492, 0.06022044088176351
0.6997350866115796, 0.07735470941883765
0.7995964531803335, 0.10771543086172344
0.8994235045433335, 0.14559118236472945];%german scale
paper_MSEs{4} = [0.1006168780999185, 0.03303085299455538
0.20011483143412037, 0.03357531760435578
0.30027329275350023, 0.0373865698729583
0.39910467839623814, 0.0373865698729583
0.4999333432308536, 0.040108892921960126
0.6001027120215483, 0.039019963702359384
0.7002793524597862, 0.03466424682395641
0.8004499331917382, 0.03303085299455538
0.899953946232226, 0.030852994555353896];  % ionosphere
to_plot = {'PE-DR','CC','ACC','Max','X','T50','PA','SPA','EM'};  % 'SCC','MS','MM',

for iData = 1:length(datasets)
    f = figure; hold on
    dataset = datasets{iData};
    data = load(['results/', dataset, '.mat']);
    methods = data.methods;
    results = data.results;
    test_pos_priors = data.test_pos_priors;
    
    col = hsv(length(to_plot));
    counter = length(to_plot);
    for iMethod = fliplr(1:size(results(1).MSEs_ci, 2))
        method = methods{iMethod};
        if strmatch(method,to_plot) ~= 0
            lower_ci = max(arrayfun(@(x) x.MSEs_ci(1, iMethod), results), 0);
            upper_ci = max(arrayfun(@(x) x.MSEs_ci(2, iMethod), results), 0);
            MSEs = arrayfun(@(x) mean(x.MSEs(:, iMethod)), results);
            if strcmp(method, 'PE-DR')
                line_style = '--';
                line_width = 2;
            else
                line_style = '-';
                line_width = 1.3;
            end
            plot(test_pos_priors, MSEs, line_style, 'color', col(counter,:), 'LineWidth', line_width); hold on
            %errorbar(test_pos_priors, MSEs, min(MSEs, MSEs-lower_ci), upper_ci-MSEs, 'color',col(iMethod,:)); hold on
            counter = counter - 1;
        end
    end
    plot(test_pos_priors, paper_MSEs{iData}(:, 2), 'r', 'LineWidth', 1.3)
    legend([fliplr(to_plot), {'du Plessis 2014'}]);
    xlabel('True class prior')
    ylabel('Mean squared error')
    title(dataset_names{iData});
    saveas(f, [dataset, '.eps'], 'epsc')
end

%% Trafficking data
f = figure;
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
saveas(f, 'trafficking.eps', 'epsc')
