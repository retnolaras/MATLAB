
    populationsize = 20;
    selthreshold = 1/populationsize;
    hiddenunitno = 39;
    maxepoch = 10;
    nnselected = 0;



err = 1.0;
load('usps_all');

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

traininput = Xtrain';
traintarget = Ytrain';
testinput = Xtest';
testtarget = Ytest';


[attrno,trainexpno] = size(traininput);         % 'attrno' is the number of attributes, 'trainexpno' is the number of training examples
[classno,n] = size(traintarget);                % 'classno' is the number of class labels, 'n' is useless
% [classno, n] = size(ylabel');

rand('state',sum(100*clock)); 


% load the validation set used in selecting the component neural networks 

[n,vexpno] = size(traininput);                  % 'vexpno' is the number of validation examples, 'n' is useless
v.P = traininput;
v.T = traintarget;
vinput = traininput;
vtarget = traintarget;




% generate the component neural networks
for i = 1:populationsize
hiddenLayerSize = 39;
net = patternnet(hiddenLayerSize);
net.trainFcn = 'trainscg';  % Scaled conjugate gradient
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
[net,tr] = train(net,traininput,traintarget);

    % save the component neural networks
    netfile = strcat('net',dec2base(i,10));
    save(netfile,'net');
end

% Test the Network
for i = 1:populationsize
            netfile = strcat('net',dec2base(i,10));
            load(netfile);                          
            output = net(testinput);
            enoutput = enoutput + output;
end
enoutput = enoutput / sum(output);

err = sum(any(xor(enoutput,testtarget))) / testexpno;           % obtain the classification error of the ensemble    
errors = gsubtract(Ytest',output);
performance = perform (net,Ytest', output);
Confmat = confusionmat(Ytest,(round(enoutput))');
figure,
heatmap(Confmat, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
title('Confusion Matrix: Neural Network')
