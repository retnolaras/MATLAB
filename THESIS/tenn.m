
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


[noattribute,notrainexp] = size(traininput);         
[noclass,n] = size(traintarget);               

rand('state',sum(100*clock)); 

% generate validation set 
novalexp = notrainexp;
valinput = zeros(noattribute,novalexp);
valtarget = zeros(noclass,novalexp);
for i = 1:novalexp
    appear = floor(rand * novalexp) + 1;
    valinput(:,i) = traininput(:,appear);
    valtarget(:,i) = traintarget(:,appear);
end
v.P = valinput;
v.T = valtarget;

% v.P = traininput;
% v.T = traintarget;
% vinput = traininput;
% vtarget = traintarget;



% generate neural networks
for i = 1:populationsize
hiddenunitno = 20;
net = newff(minmax(traininput),[hiddenunitno noclass],{'tansig' 'purelin'});
net.trainFcn = 'trainscg';  
net.performFcn = 'mse'; 

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

net.efficiency.memoryReduction = 100;
net.trainParam.max_fail = 6;
net.trainParam.min_grad=1e-5;
net.trainParam.show=10;
net.trainParam.lr=0.9;
net.trainParam.epochs=100;
net.trainParam.goal=0.00;

% Train the Network
net = train(net,traininput,traintarget,[],[],v);
netfile = strcat('net',dec2base(i,10));
save(netfile,'net');
end

% generate the ensemble
tic;
valoutput = [];
    for i = 1:populationsize
        netfile = strcat('net',dec2base(i,10));
        load(netfile);
        output = sim(net,valinput);                               
        outputbool = (output == repmat(max(output),noclass,1));    
        valoutput = [valoutput;outputbool];
    end
    bounds = repmat([0 1],populationsize,1);
    best = ga(bounds,'evalenn',[valoutput(:)',valtarget(:)',populationsize,selthreshold,noclass,novalexp]); 

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
[n,notestexp] = size(testinput);
ennoutput = zeros(noclass,notestexp);                            
    for i = 1:populationsize
        if selected(i) == 1                                         % if the i-th component neural network was selected
            netfile = strcat('net',dec2base(i,10));
            load(netfile);
            output = sim(net,testinput);
            outputbool = (output==repmat(max(output),noclass,1));
            ennoutput = ennoutput + outputbool;                           % sum the votes for the class labels
            nnselected = nnselected +1;
        end
    end
    
ennoutput = (ennoutput == repmat(max(ennoutput),noclass,1));       % now 'enoutput' stores the boolean output of the ensemble,
err = sum(any(xor(ennoutput,testtarget))) / notestexp; 
end

