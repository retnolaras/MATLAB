
populationsize = 20;
selthreshold = 1/populationsize;
nnselected = 0;
err = 1.0;
load('usps_all');

reduce_dim = false;
X = double(reshape(data,256,11000)');
ylabel = [1:9 0];
y = reshape(repmat(ylabel,1100,1),11000,1);

clearvars data

cv = cvpartition(y, 'kfold', 10);
%KFolds partition data to train and test
for j = 1:cv.NumTestSets
Xtrain = X(cv.training(j),:);
Ytrain = y(cv.training(j),1);
Xtest = X(cv.test(j),:);
Ytest = y(cv.test(j),1); 

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



% generate the component neural networks
for i = 1:populationsize
hiddenLayerSize = 30;
net = patternnet(hiddenLayerSize);
net.trainFcn = 'trainscg';  % Scaled conjugate gradient
net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

net.efficiency.memoryReduction = 100;
% net.trainParam.max_fail = 6;
% net.trainParam.min_grad=1e-5;
% net.trainParam.show=10;
% net.trainParam.lr=0.9;
net.trainParam.epochs=1000;
% net.trainParam.epochs=100;
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
%     enoutput = zeros(classno,testexpno);                            % 'enoutput' is the test result of the ensemble
enoutput = zeros(classno,trainexpno);
    for i = 1:populationsize
        if selected(i) == 1                                         % if the i-th component neural network was selected
            netfile = strcat('net',dec2base(i,10));
            load(netfile);
%             output = sim(net,testinput);                            % now 'output' stores the real-valued output of the component neural network
%             output = net(testinput);
output=net(traininput);
%             output = (output == repmat(max(output),classno,1));     % now 'output' stores the boolean output of the component neural networks, 
                                                                    % in other words, a winner-take-all competition has been performed                                                         
            enoutput = enoutput + output;                           % sum the votes for the class labels
            nnselected = nnselected +1;

        end
    end
    enoutput = enoutput / sum(selected); 
%   enoutput = (enoutput == repmat(max(enoutput),classno,1));       % now 'enoutput' stores the boolean output of the ensemble,
                                                                    % the class label receiving the most number of votes is '1', others are '0's
% err = sum(any(xor(enoutput,testtarget))) / testexpno;           % obtain the classification error of the ensemble    
% errors = gsubtract(Ytest',output);
% performance = perform (net,Ytest', output);
% err = immse(Ytest,enoutput');
% I=(round(enoutput))';
% disp(I);
% fid = fopen('output.txt','a');
% if (I==1)
%     fprintf(fid,'1');
% fclose(fid);
% elseif (I==2)
%     fprintf(fid,'2');
% fclose(fid);     
% elseif (I==3)
%     fprintf(fid,'3');
% fclose(fid);     
% elseif (I==4)
%     fprintf(fid,'4');
% fclose(fid);     
% elseif (I==5)
%     fprintf(fid,'5');
% fclose(fid);     
% elseif (I==6)
%     fprintf(fid,'6');
% fclose(fid);     
% elseif (I==7)
%     fprintf(fid,'7');
% fclose(fid);     
% elseif (I==8)
%     fprintf(fid,'8');
% fclose(fid);     
% elseif (I==9)
%     fprintf(fid,'9');
% fclose(fid);     
% elseif (I==10)
%     fprintf(fid,'10');
% fclose(fid);     
%  
%  else
%      disp(' not Found');
% end


% Confmat = confusionmat(Ytest,(round(enoutput))');
Confmat = confusionmat(Ytrain,(round(enoutput))');
figure,
heatmap(Confmat, 0:9, 0:9, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
title('Confusion Matrix: Ensemble Neural Network')
end
