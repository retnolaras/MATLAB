disp('testing')
load ('featureout.mat');
% p=reshape(featureout,1,[]);
fo= featureout';
p=featureout(1,:);

populationsize = 20;
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
load net.mat;
load net;
% load selected.mat;

% load(testset);
% testexpno = 1;
% enoutput = zeros(1,1);                            % 'enoutput' is the test result of the ensemble
%     for i = 1:populationsize
%         if selected(i) == 1                                         % if the i-th component neural network was selected
%             netfile = strcat('net',dec2base(i,10));
%             load(netfile);
%             output_ = sim(net,p);                            % now 'output' stores the real-valued output of the component neural network
%             enoutput = enoutput + output_;                           % sum the votes for the class label
%         end
%     end
%     enoutput = enoutput / sum(selected); 
% 
enoutput=sim(net,p);

fid = fopen('output.txt','a');
I=round(enoutput);
  
if (I==1)
    fprintf(fid,'1');
fclose(fid);
elseif (I==2)
    fprintf(fid,'2');
fclose(fid);     
elseif (I==3)
    fprintf(fid,'3 ');
fclose(fid);     
elseif (I==4)
    fprintf(fid,'4');
fclose(fid);     
elseif (I==5)
    fprintf(fid,'5');
fclose(fid);     
elseif (I==6)
    fprintf(fid,'6');
fclose(fid);     
elseif (I==7)
    fprintf(fid,'7');
fclose(fid);     
elseif (I==8)
    fprintf(fid,'8');
fclose(fid);     
elseif (I==9)
    fprintf(fid,'9');
fclose(fid);     
elseif (I==0)
    fprintf(fid,'0');
fclose(fid);     
else
     disp(' not Found');
     
   clear
end
 
 
 