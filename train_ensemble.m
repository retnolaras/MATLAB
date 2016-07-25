
    populationsize = 20;
    selthreshold = 1/populationsize;
    hiddenunitno = 39;
    maxepoch = 10;
    nnselected = 0;



err = 1.0;
% load(trainset);
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
% load(validset);
[n,vexpno] = size(traininput);                  % 'vexpno' is the number of validation examples, 'n' is useless
v.P = traininput;
v.T = traintarget;
vinput = traininput;
vtarget = traintarget;


% generate a validation set via bootstrap sampling from the training set, which is used in training the component neural networks.
% note that this validation set is different from the "validset". If plentiful training data are available, it will be better 
% to use a seperate validation set for this purposes. In our previous experiments, good performance is achieved with out-of-bag 
% validation. About out-of-bag validation, please refer L. Breiman's work "Out-of-bag estimation".
% vexpno = trainexpno;                       % the size of the validation set
% vinput = zeros(attrno,vexpno);
% vtarget = zeros(classno,vexpno);
% for i = 1:vexpno
%     appear = floor(rand * vexpno) + 1;     % 'appear' indicates which example should appear in the validation set
%     vinput(:,i) = traininput(:,appear);
%     vtarget(:,i) = traintarget(:,appear);
% end
% v2.P = vinput;
% v2.T = vtarget;


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
% net.trainParam.epochs=1000;
net.trainParam.epochs=100;
net.trainParam.goal=0.00;


% Train the Network
[net,tr] = train(net,traininput,traintarget);

    % save the component neural networks
    netfile = strcat('net',dec2base(i,10));
    save(netfile,'net');
end


% generate the ensemble
tic;
% if classno > 1                                                  % classification task
    voutput = [];
    for i = 1:populationsize
        netfile = strcat('net',dec2base(i,10));
        load(netfile);
        output = sim(net,vinput);                               % now 'output' stores the real-valued output of the component neural network
        output = (output == repmat(max(output),classno,1));     % now 'output' stores the boolean output of the component neural networks,
                                                                % in other words, a winner-take-all competition has been performed
        voutput = [voutput;output];
    end
    bounds = repmat([0 1],populationsize,1);
    best = ga(bounds,'gasenEvalC',[voutput(:)',vtarget(:)',populationsize,selthreshold,classno,vexpno]); 
                                                                % 'best' is the evolved best weight vector
                                                                
% else                                                            % regression task
%     voutput = [];
%     cor = zeros(populationsize);                                % correlation matrix defined in the AIJ paper
%     error = zeros(populationsize,vexpno); 
%     for i = 1:populationsize
%         netfile = strcat('net',dec2base(i,10));
%         load(netfile);
%         voutput = sim(net,vinput);
%         error(i,:) = voutput - vtarget;
%     end
%     for i = 1:populationsize
%         for j = 1:populationsize
%             for k = 1:vexpno
%                 cor(i,j) = cor(i,j) + error(i,k) * error(j,k);
%             end
%             cor(i,j) = cor(i,j) / vexpno;
%         end
%     end
%     bounds = repmat([0 1],populationsize,1);
%     best = ga(bounds,'gasenEvalR',[cor(:)',selthreshold]);
% end
toc;


% create a file 'selected.mat' to record the index of the selected component neural networks
best(length(best)) = [];                                            % see gademo of GAOT toolbox
best = best./sum(best);
selected = zeros(1,populationsize);
for i = 1:populationsize
    if best(i) >= selthreshold
        selected(i) = 1;
    end
end
save('selected','selected');


% test the ensemble
% load(testset);
[n,testexpno] = size(testinput);                                    % 'testexpno' is the number of test examples, 'n' is useless
% if classno > 1                                                      % classification task
    enoutput = zeros(classno,testexpno);                            % 'enoutput' is the test result of the ensemble
    for i = 1:populationsize
        if selected(i) == 1                                         % if the i-th component neural network was selected
            netfile = strcat('net',dec2base(i,10));
            load(netfile);
%             output = sim(net,testinput);                            % now 'output' stores the real-valued output of the component neural network
            output = net(testinput);
%             output = (output == repmat(max(output),classno,1));     % now 'output' stores the boolean output of the component neural networks, 
                                                                    % in other words, a winner-take-all competition has been performed
                                                            
            enoutput = enoutput + output;                           % sum the votes for the class labels
            nnselected = nnselected +1;
            


        end
    end
    enoutput = enoutput / sum(selected); 
%     enoutput = (enoutput == repmat(max(enoutput),classno,1));       % now 'enoutput' stores the boolean output of the ensemble,
                                                                    % the class label receiving the most number of votes is '1', others are '0's
    err = sum(any(xor(enoutput,testtarget))) / testexpno;           % obtain the classification error of the ensemble    
% else                                                                % regression task
%     enoutput = zeros(classno,testexpno);                            % 'enoutput' is the test result of the ensemble
%     for i = 1:populationsize
%         if selected(i) == 1                                         % if the i-th component neural network was selected
%             netfile = strcat('net',dec2base(i,10));
%             load(netfile);
%             output = sim(net,testinput);                            % now 'output' stores the real-valued output of the component neural network
%             enoutput = enoutput + output;                           % sum the real-valued outputs
%         end
%     end
%     enoutput = enoutput / sum(selected);                            % now 'enoutput' stores the output of the ensemble
%     err = mse(enoutput - testtarget);                               % obtain the mean squared error of the ensemble    
% end
errors = gsubtract(Ytest',output);
performance = perform (net,Ytest', output);
Confmat = confusionmat(Ytest,(round(enoutput))');
figure,
heatmap(Confmat, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
title('Confusion Matrix: Neural Network')
