function [trainx, trainy, testx, testy] = getSyntheticData()
randseed(1);

% Parameters
ntrain = 400;
ntest = 400;
py_train = 0.5;
py_test = 0.05;

% Training data
trainy = rand([ntrain,1]) < py_train; % proportion of 1's ~= py_train
trainx = [randn([ntrain,1]) + trainy*10 randn([ntrain,1]) + trainy*10]; % mean = 0 if y=0; 10 if y=1

idxs = randperm(length(trainy));
trainy = trainy(idxs);
trainx = trainx(idxs,:);

% Testing data
testy = rand([ntest,1]) < py_test; % proportion of 1's ~= py_train
testx = [randn([ntest,1]) + testy*10 randn([ntest,1]) + testy*10]; % mean = 0 if y=0; 10 if y=1

idxs = randperm(length(testy));
testy = testy(idxs);
testx = testx(idxs,:);

% Display data
figure(1);
clf; subplot(1,2,1)
scatter(trainx(trainy==0,1),trainx(trainy==0,2),'b')
hold on; scatter(trainx(trainy==1,1),trainx(trainy==1,2),'r');
legend({'y=0','y=1'}); axis square;title('Training data'); 
xlabel('x1'); ylabel('x2')
hold off

subplot(1,2,2);
scatter(testx(testy==0,1),testx(testy==0,2),'b')
hold on; scatter(testx(testy==1,1),testx(testy==1,2),'r');
legend({'y=0','y=1'}); axis square;
title('Testing data')
xlabel('x1'); ylabel('x2')
hold off


end
