load('usps_all');

reduce_dim = false;
X = double(reshape(data,256,11000)');
ylabel = [1:9 0];

y = reshape(repmat(ylabel,1100,1),11000,1);

clearvars data

cv = cvpartition(y, 'kfold', 10);
%KFolds partition data to train and test
for i = 1:cv.NumTestSets
Xtrain = X(cv.training(i),:);
Ytrain = y(cv.training(i),1);
Xtest = X(cv.test(i),:);
Ytest = y(cv.test(i),1); 


[A,Ascore,Alatent,xx,Aexplain] = pca(Xtrain');
[B,Bscore,Blatent,xx] = pca(Xtest'); 

net = patternnet(39);
net.trainFcn = 'trainscg';  % Scaled conjugate gradient
net.performFcn = 'mse';  % Mean squared error
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

net.efficiency.memoryReduction = 100;
net.trainParam.max_fail = 6;
net.trainParam.min_grad=1e-5;
net.trainParam.show=10;
net.trainParam.lr=0.9;
net.trainParam.epochs=1000;
net.trainParam.goal=0.00;
[net,tr] = train(net,A(:,1:50)',Ytrain');
outputs = net(B(:,1:50)');
errors = gsubtract(Ytest',outputs);
performance = perform (net,Ytest', outputs);
cp= classperf(Ytest,round(outputs'));
end

save net;

% Confmat = confusionmat(Ytest,round(outputs'));
% figure,
% heatmap(Confmat, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
% title('Confusion Matrix: Neural Network')