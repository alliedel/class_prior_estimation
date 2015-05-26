files = dir('insurance*.mat');

for i = 1:length(files)
    file = files(i).name;
    name = file(1:end-4);
    data = load(file);
    test_pos_priors = data.test_pos_priors;
    results = data.results;
    methods = data.methods;
    priors = cell2mat(arrayfun(@(x) mean(x.priors), results, 'UniformOutput', false)');
    MSEs = cell2mat(arrayfun(@(x) mean(x.MSEs), results, 'UniformOutput', false)');
    RAEs = cell2mat(arrayfun(@(x) mean(x.RAEs), results, 'UniformOutput', false)');
    KLDs = cell2mat(arrayfun(@(x) mean(x.KLDs), results, 'UniformOutput', false)');
    xlswrite('all.xls', priors, [name, '_priors'], 'B2');
    xlswrite('all.xls',test_pos_priors(:), [name, '_priors'], 'A2');
    xlswrite('all.xls',methods(:)', [name, '_priors'], 'B1');
    
    xlswrite('all.xls', MSEs, [name, '_MSE'], 'B2');
    xlswrite('all.xls', RAEs, [name, 'RAE'], 'B2');
    xlswrite('all.xls', KLDs, [name, '_KLD'], 'B2');
    
end
    