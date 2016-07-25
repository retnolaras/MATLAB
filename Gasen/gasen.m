function [err] = gasen(trainset,testset,validset,populationsize,selthreshold,hiddenunitno,maxepoch)
%
% use GASEN to create an ensemble of BP networks
% function [err] = gasen(trainset,testset,validset,populationsize,selthreshold,hiddenunitno,maxepoch)
%
% to use this function, GAOT toolbox must be available. refer: C.R. Houck, J.A. Joines, and M.G. Kay. A genetic algorithm for 
% function optimization: a Matlab implementation, Technical Report: NCSU-IE-TR-95-09, North Carolina State University, Raleigh, 
% NC, 1995.
%
% ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact 
% Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
%
% trainset           -       file path for training set, e.g. 'C:\gasen\train.mat', the file should contain two parts, 
%                            i.e. traininput and traintarget.
%
%                            For classification, traintarget is a class matrix where each class label occupys a row, 
%                            each column is an output vector, where only the element representing the class of current example is '1'.
%
%                            for example, the traininput and traintarget for XOR problem are respective:
%                                             0  0  1  1     0  1  1  0
%                                             0  1  0  1     1  0  0  1
%
%                            For regression, traintarget is a vector where each element is the a real-valued output.
%                            Here the code deals with 1-output regression tasks. It judges whether the task is a classification
%                            one or regression one through counting the rows of traintarget.
%
% testset            -       file path for test set, e.g. 'C:\gasen\test.mat', the file should contain two parts, 
%                            i.e. testinput and testtarget.
%
%                            For classification, traintarget is a class matrix where each class label occupys a row, 
%                            each column is an output vector, where only the element representing the class of current example is '1'.
%
%                            For regression, traintarget is a vector where each element is the a real-valued output.
%                            Here the code deals with 1-output regression tasks.
%
% validset           -       file path for validation set used in selecting the component networks, e.g. 'C:\gasen\valid.mat', the 
%                            file should contain two parts, i.e. vinput and vtarget.
%
%                            For classification, vtarget is a class matrix where each class label occupys a row, 
%                            each column is an output vector, where only the element representing the class of current example is '1'.
%
%                            For regression, vtarget is a vector where each element is the a real-valued output.
%                            Here the code deals with 1-output regression tasks.
%
%                            It will be better to use a seperate validation set. However, if the data is not plentiful, a possible
%                            choice is to bootstrap sample a validation set from the training set.
%
% populationsize     -       number of neural networks in the population, default value is 20
%
% selthreshold       -       threshold for selecting the component neural networks, default value is 1/populationsize
%
% hiddenunitno       -       number of hidden units of the neural networks, default value is 5
%
% maxepoch           -       maximum training epochs of the component neural networks, default value is 100
%
% err                -       test set error of the ensemble
%
%
% Reference: Z.-H. Zhou, J. Wu, and W. Tang. Ensembling neural networks: Many could be better than all. 
%            Artificial Intelligence, 2002, 137(1-2): 239-263.
%
% ATTN2: This package was developed by Mr. Wei Tang (tangwei@ai.nju.edu.cn). For any problem concerning the code,
% please feel free to contact Mr. Tang.
%
%

if nargin == 3
    populationsize = 20;
    selthreshold = 1/populationsize;
    hiddenunitno = 5;
    maxepoch = 100;
elseif nargin == 4
    selthreshold = 1/populationsize;
    hiddenunitno = 5;
    maxepoch = 100;
elseif nargin == 5
    hiddenunitno = 5;
    maxepoch = 100;
elseif nargin == 6
    maxepoch = 100;
end


err = 1.0;
load(trainset);


[attrno,trainexpno] = size(traininput);         % 'attrno' is the number of attributes, 'trainexpno' is the number of training examples
[classno,n] = size(traintarget);                % 'classno' is the number of class labels, 'n' is useless


rand('state',sum(100*clock)); 


% load the validation set used in selecting the component neural networks 
load(validset);
[n,vexpno] = size(vinput);                  % 'vexpno' is the number of validation examples, 'n' is useless
v.P = vinput;
v.T = vtarget;


% generate a validation set via bootstrap sampling from the training set, which is used in training the component neural networks.
% note that this validation set is different from the "validset". If plentiful training data are available, it will be better 
% to use a seperate validation set for this purposes. In our previous experiments, good performance is achieved with out-of-bag 
% validation. About out-of-bag validation, please refer L. Breiman's work "Out-of-bag estimation".
v2expno = trainexpno;                       % the size of the validation set
v2input = zeros(attrno,v2expno);
v2target = zeros(classno,v2expno);
for i = 1:v2expno
    appear = floor(rand * v2expno) + 1;     % 'appear' indicates which example should appear in the validation set
    v2input(:,i) = traininput(:,appear);
    v2target(:,i) = traintarget(:,appear);
end
v2.P = v2input;
v2.T = v2target;


% generate the component neural networks
for i = 1:populationsize
    % generate the component training sets
    compinput = zeros(attrno,trainexpno); 
    comptarget = zeros(classno,trainexpno);
    for j = 1:trainexpno
        appear = floor(rand * trainexpno) + 1;       % 'appear' indicates which example should appear in current component training set
        compinput(:,j) = traininput(:,appear);
        comptarget(:,j) = traintarget(:,appear);
    end
    
    % train the component neural networks
    net = newff(MinMax(compinput),[hiddenunitno classno],{'tansig' 'purelin'});
    net.trainParam.epochs = maxepoch;
    net.trainParam.goal = 0.0;
    net = train(net,compinput,comptarget,[],[],v2);

    % save the component neural networks
    netfile = strcat('net',dec2base(i,10));
    save(netfile,'net');
end


% generate the ensemble
tic;
if classno > 1                                                  % classification task
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
                                                                
else                                                            % regression task
    voutput = [];
    cor = zeros(populationsize);                                % correlation matrix defined in the AIJ paper
    error = zeros(populationsize,vexpno); 
    for i = 1:populationsize
        netfile = strcat('net',dec2base(i,10));
        load(netfile);
        voutput = sim(net,vinput);
        error(i,:) = voutput - vtarget;
    end
    for i = 1:populationsize
        for j = 1:populationsize
            for k = 1:vexpno
                cor(i,j) = cor(i,j) + error(i,k) * error(j,k);
            end
            cor(i,j) = cor(i,j) / vexpno;
        end
    end
    bounds = repmat([0 1],populationsize,1);
    best = ga(bounds,'gasenEvalR',[cor(:)',selthreshold]);
end
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
load(testset);
[n,testexpno] = size(testinput);                                    % 'testexpno' is the number of test examples, 'n' is useless
if classno > 1                                                      % classification task
    enoutput = zeros(classno,testexpno);                            % 'enoutput' is the test result of the ensemble
    for i = 1:populationsize
        if selected(i) == 1                                         % if the i-th component neural network was selected
            netfile = strcat('net',dec2base(i,10));
            load(netfile);
            output = sim(net,testinput);                            % now 'output' stores the real-valued output of the component neural network
            output = (output == repmat(max(output),classno,1));     % now 'output' stores the boolean output of the component neural networks, 
                                                                    % in other words, a winner-take-all competition has been performed
                                                            
            enoutput = enoutput + output;                           % sum the votes for the class labels
        end
    end
    enoutput = (enoutput == repmat(max(enoutput),classno,1));       % now 'enoutput' stores the boolean output of the ensemble,
                                                                    % the class label receiving the most number of votes is '1', others are '0's
    err = sum(any(xor(enoutput,testtarget))) / testexpno;           % obtain the classification error of the ensemble    
else                                                                % regression task
    enoutput = zeros(classno,testexpno);                            % 'enoutput' is the test result of the ensemble
    for i = 1:populationsize
        if selected(i) == 1                                         % if the i-th component neural network was selected
            netfile = strcat('net',dec2base(i,10));
            load(netfile);
            output = sim(net,testinput);                            % now 'output' stores the real-valued output of the component neural network
            enoutput = enoutput + output;                           % sum the real-valued outputs
        end
    end
    enoutput = enoutput / sum(selected);                            % now 'enoutput' stores the output of the ensemble
    err = mse(enoutput - testtarget);                               % obtain the mean squared error of the ensemble    
end

% end of function

