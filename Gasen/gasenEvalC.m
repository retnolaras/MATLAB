function [sol,val] = gasenEvalC(sol,options)
%
% fitness function used by GASEN for classification
%
% to use this function, GAOT toolbox must be available. refer: C.R. Houck, J.A. Joines, and M.G. Kay. A genetic algorithm for 
% function optimization: a Matlab implementation, Technical Report: NCSU-IE-TR-95-09, North Carolina State University, Raleigh, 
% NC, 1995.
%
% ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact 
% Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
%
% see gademo of GAOT toolbox for the meaning of 'sol', 'val', and 'options'
%
% Reference: Z.-H. Zhou, J. Wu, and W. Tang. Ensembling neural networks: Many could be better than all. 
%            Artificial Intelligence, 2002, 137(1-2): 239-263.
%
% ATTN2: This package was developed by Mr. Wei Tang (tangwei@ai.nju.edu.cn). For any problem concerning the code,
% please feel free to contact Mr. Tang.
%
%

populationsize = options(length(options) - 3);      % number of neural networks in the population
selthreshold = options(length(options) - 2);        % threshold for selecting the component neural networks
classno = options(length(options) - 1);             % number of class labels
vexpno = options(length(options));                  % number of validation examples


% obtain the validation output of the component neural networks and the target of the validation set
options = options(1 : length(options) - 4);

voutput = options(2 : populationsize * classno * vexpno + 1);
voutput = reshape(voutput,populationsize * classno,vexpno);

vtarget = options(populationsize * classno * vexpno + 2 : length(options));
vtarget = reshape(vtarget,classno,vexpno);


x = sol(1:populationsize);                          % 'x' is un-normalized individual
x = x ./ sum(x);
sol(1:populationsize) = x;                          % now the individual has been normalized


% compute the fitness
enoutput = zeros(classno,vexpno);
for i = 1:populationsize
    if x(i) >= selthreshold
        enoutput = enoutput + voutput((i - 1) * classno + 1 : i * classno, :);
    end
end
enoutput = (enoutput == repmat(max(enoutput),classno,1));
if sum(any(xor(enoutput,vtarget)))
    val = populationsize / sum(any(xor(enoutput,vtarget)));
else
    val = inf;
end


% end of function
