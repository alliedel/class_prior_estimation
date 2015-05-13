function Fnc = competitors(X_train, y_train, X_test)

  % train calibrated model
  classes = sort(unique(y_train));
  if classes(1) == 0
      y_train = y_train + 1;
      classes = classes+1;
  elseif classes(1) == -1
      classes = (classes+1)/2+1;
      y_train = (y_train+1)/2+1;
  end
  betas = glmfit(X_train,y_train-1,'binomial','link','logit');
  %betas = mnrfit(X_train,y_train);
  %pos = mnrval(betas, X_train(y_train==classes(2), :));
  pos = glmval(betas, X_train(y_train==classes(2), :), 'logit');
  %pos = pos(:,2);  % positive class is second column
  %neg = mnrval(betas, X_train(y_train==classes(1), :));
  neg = glmval(betas, X_train(y_train==classes(1), :), 'logit');
  %neg = neg(:,2);  % positive class is secon column
  %test = mnrval(betas, X_test);
  test = glmval(betas, X_test, 'logit');
  %test = test(:,2);  % positive class is secon column

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Methods due to Forman 2005 %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %get thresholds for Forman's acc based methods
  Acc = @(t) -(sum(pos>=t)+sum(neg<t))/(size(pos,1)+size(neg,1));
  Max = @(t) -(mean(pos>=t)-mean(neg>=t));
  X = @(t) abs(mean(pos>=t)+mean(neg>=t)-1);
  T50 = @(t) abs(mean(pos>=t)-0.5);
  Acc_thresh = fminsearch(Acc,mean(pos));
  Max_thresh = fminsearch(Max,mean(pos));
  X_thresh = fminsearch(X,mean(pos));
  T50_thresh = fminsearch(T50,mean(pos));
  
  %method - CC
  thresh = Acc_thresh;
  prev_CC = mean(test>=thresh);
  
  %method - ACC
  thresh = Acc_thresh;
  prev_ACC = (mean(test>=thresh)-mean(neg>=thresh))/(mean(pos>=thresh)-mean(neg>=thresh));
  
  %method - Max
  thresh = Max_thresh;
  prev_Max = (mean(test>=thresh)-mean(neg>=thresh))/(mean(pos>=thresh)-mean(neg>=thresh));
  
  %method - X
  thresh = X_thresh;
  prev_X = (mean(test>=thresh)-mean(neg>=thresh))/(mean(pos>=thresh)-mean(neg>=thresh));
  
  %method - T50
  thresh = T50_thresh;
  prev_T50 = (mean(test>=thresh)-mean(neg>=thresh))/(mean(pos>=thresh)-mean(neg>=thresh));
  
  %method - MS
  threshs = unique([neg;pos]);
  threshs = datasample(threshs,min(size(threshs,1),500),'Replace',false);
  prev_MS = zeros(size(threshs,1),1);
  for i = 1:size(threshs,1)
      thresh = threshs(i);
      prev_MS(i) = (mean(test>=thresh)-mean(neg>=thresh))/(mean(pos>=thresh)-mean(neg>=thresh));
  end
  prev_MS = median(prev_MS);
  
  %method - MM
  if any(isnan(pos))
      print 'Invalid estimate'
  end
  [cdf_pos,x_pos] = ecdf(pos);
  [cdf_neg,x_neg] = ecdf(neg);
  [cdf_test,x_test] = ecdf(test);
  x = linspace(0,1,1000);
  if x_pos(1) == x_pos(2)  % first and second entries are the same
    x_pos(1) = [];
    cdf_pos(1) = [];  
    x_neg(1) = [];
    cdf_neg(1) = [];
    x_test(1) = [];
    cdf_test(1) = [];
  end
  cdf_pos = interp1(x_pos,cdf_pos,x);
  cdf_neg = interp1(x_neg,cdf_neg,x);
  cdf_test = interp1(x_test,cdf_test,x);
  pp_metric = @(p) trapz(abs(cdf_test-(p*cdf_pos-(1-p)*cdf_neg)));
  prev_MM = fminbnd(pp_metric,0,1);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Methods due to Bella 2010  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  % method - PA (a straw man)
  prev_PA = mean(test);
  % method - SPA
  prev_SPA = (mean(test)-mean(neg))/(mean(pos)-mean(neg));
  % method - SCC
  prev_SCC = (mean(test>=Acc_thresh)-mean(neg))/(mean(pos)-mean(neg));
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Methods due to Saerens 2002%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % method - EM
  tol = 1e-20;
  prev_tr = size(pos,1)/(size(pos,1)+size(neg,1));
  prev_EM = prev_tr;
  for i = 1:200
    last_prev = prev_EM;
    prob = prev_EM./prev_tr.*test ./ (prev_EM./prev_tr.*test + (1-prev_EM)./(1-prev_tr).*(1-test));
    prev_EM = mean(prob);
    if abs(prev_EM-last_prev)<tol
        break
    end
  end
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % other methods %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Esuli and Sebastiani, 2013 (SVMperf)
  % Milli et al. 2013 (quantification trees & forests)

  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%  
  
  Fnc = [prev_CC,prev_ACC,prev_Max,prev_X,prev_T50,prev_MS,prev_MM,prev_PA,prev_SPA,prev_SCC,prev_EM];
  Fnc(isnan(Fnc)) = 0;
end
  