load('usps_all');

reduce_dim = false;
X = double(reshape(data,256,11000)');
ylabel = [1:9 0];

y = reshape(repmat(ylabel,1100,1),11000,1);

clearvars data



cv = cvpartition(y, 'kfold', 10);
% KFolds partition data to train and test
for i = 1:cv.NumTestSets
Xtrain = X(cv.training(i),:);
Ytrain = y(cv.training(i),1);
Xtest = X(cv.test(i),:);
Ytest = y(cv.test(i),1); 



% cv = cvpartition(y, 'holdout', .5);
% Xtrain = X(cv.training,:);
% Ytrain = y(cv.training,1);
% 
% Xtest = X(cv.test,:);
% Ytest = y(cv.test,1);

% TRAIN
% mdl_ctree = ClassificationTree.fit(Xtrain,Ytrain);
% ypred = predict(mdl_ctree,Xtest);
% Confmat_ctree = confusionmat(Ytest,ypred);

mdl = fitensemble(Xtrain,Ytrain,'bag',200,'tree','type','Classification');
% ypred = predict(mdl,Xtest);
ypred = predict(mdl,Xtrain);
% Confmat_bag = confusionmat(Ytest,ypred);
Confmat_bag = confusionmat(Ytrain,ypred);

% cpname1 = strcat('cp1',dec2base(i,10));
% cp= classperf(Ytest,ypred);
% save(cpname1,'cp');

net = patternnet(39);
net.trainFcn = 'trainscg';  % Scaled conjugate gradient
net.performFcn = 'mse';  % Mean squared error
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};


net.efficiency.memoryReduction = 100;
% net.trainParam.max_fail = 6;
% net.trainParam.min_grad=1e-5;
% net.trainParam.show=10;
% net.trainParam.lr=0.9;
net.trainParam.epochs=1000;
% net.trainParam.goal=0.00;
[net,tr] = train(net,Xtrain',Ytrain');
% outputs = net(Xtest');
outputs= net(Xtrain');
% errors = gsubtract(Ytest',outputs);
% performance = perform (net,Ytest', outputs);
% cp2= classperf(Ytest,round(outputs'));
% cpname2 = strcat('cp2',dec2base(i,10));
% save(cpname2,'cp2');

% err = immse(Ytest,outputs');

% save net;

% Confmat = confusionmat(Ytest,(round(outputs))');
Confmat = confusionmat(Ytrain,(round(outputs))');
% 


%  CONFMAT
% figure,
% heatmap(Confmat_ctree, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
% title('Confusion Matrix: Single Classification Tree')
figure,
heatmap(Confmat_bag, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
title('Confusion Matrix: Ensemble of Bagged Classification Trees')

figure,
heatmap(Confmat, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
title('Confusion Matrix: Neural Network')
end