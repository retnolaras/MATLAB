function [sol,val] = gasenEvalR(sol,options)
%
% fitness function used by GASEN for regression
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

selthreshold = options(length(options));            % threshold for selecting the component neural networks

options = options(1 : length(options) - 1);


% obtain the number of neural networks in the population
n = size(options);                              
populationsize2 = n(2);                             % (square of the number of neural networks in the population) plus one
populationsize = sqrt(populationsize2 - 1);


% obtain the correlation matrix defined in the AIJ paper
cor = options(2:populationsize2);
cor = reshape(cor,populationsize,populationsize);


x = sol(1:populationsize);                          % 'x' is un-normalized individual
x = x ./ sum(x);
sol(1:populationsize) = x;                          % now the individual has been normalized


% compute the fitness
val = x * cor * x';
val = 1 / val;


% end of function
