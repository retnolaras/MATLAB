% Solve a Pattern Recognition Problem with a Neural Network


rng('default');

load('usps_all');
% load('mnist_all');

reduce_dim = false;
X = double(reshape(data,256,11000)');
ylabel = [1:9 0];

y = reshape(repmat(ylabel,1100,1),11000,1);

clearvars data

% cv = cvpartition(y, 'kfold', 10);
% %KFolds partition data to train and test
% for i = 1:cv.NumTestSets
% Xtrain = X(cv.training(i),:);
% Ytrain = y(cv.training(i),1);
% Xtest = X(cv.test(i),:);
% Ytest = y(cv.test(i),1); 
% end

cv = cvpartition(y, 'holdout', .5);
Xtrain = X(cv.training,:);
Ytrain = y(cv.training,1);

Xtest = X(cv.test,:);
Ytest = y(cv.test,1);

inputs = Xtrain';
targets = Ytrain';


% Create a Pattern Recognition Network
hiddenLayerSize = 39;
net = patternnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
%net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
%net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
%net.divideFcn = 'dividerand';  % Divide data randomly
%net.divideMode = 'sample';  % Divide up every sample
%net.divideParam.trainRatio = 80/100;
%net.divideParam.testRatio = 20/100;

% For help on training function 'trainscg' type: help trainscg
% For a list of all training functions type: help nntrain
net.trainFcn = 'trainscg';  % Scaled conjugate gradient

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

net.efficiency.memoryReduction = 100;
net.trainParam.max_fail = 6;
net.trainParam.min_grad=1e-5;
net.trainParam.show=10;
net.trainParam.lr=0.9;
net.trainParam.epochs=1000;
net.trainParam.goal=0.00;


% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
%outputs = net(inputs);
%errors = gsubtract(targets,outputs);
%performance = perform(net,targets,outputs)
outputs = net(Xtest');
errors = gsubtract(Ytest',outputs);
performance = perform (net,Ytest', outputs);

Confmat = confusionmat(Ytest,(round(outputs))');

% Recalculate Training, Validation and Test Performance
% trainTargets = targets .* tr.trainMask{1};
% valTargets = targets  .* tr.valMask{1};
% testTargets = targets  .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,outputs)
% valPerformance = perform(net,valTargets,outputs)
% testPerformance = perform(net,testTargets,outputs)

% View the Network
view(net)


% disp('after training')
% y1 = sim(net,inputs);
% y1=abs(y1);
% y1=round(y1);
% 
% save   d:\training_set\net net;

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotconfusion(targets,outputs)
%figure, plotroc(targets,outputs)
%figure, ploterrhist(errors)
figure,
heatmap(Confmat, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
title('Confusion Matrix: Neural Network')


