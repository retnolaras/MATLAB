disp('testing')
load ('featureout.mat');
p=featureout;

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
load net.mat;
load net;
load selected.mat;

load(testset);
[n,testexpno] = size(testinput);
enoutput = zeros(classno,testexpno);                            % 'enoutput' is the test result of the ensemble
    for i = 1:populationsize
        if selected(i) == 1                                         % if the i-th component neural network was selected
            netfile = strcat('net',dec2base(i,10));
            load(netfile);
            output = sim(net,p);                            % now 'output' stores the real-valued output of the component neural network
            enoutput = enoutput + output;                           % sum the votes for the class labels
            nnselected = nnselected +1;

        end
    end
    enoutput = enoutput / sum(selected); 


% y5=sim(net,p);

 disp(y5);
 [C I]=max(y5);
 disp(I)
 disp(C)
 
fid = fopen('output.txt','a');

  
if (I==1)
    fprintf(fid,'A');
fclose(fid);
elseif (I==2)
    fprintf(fid,'B');
fclose(fid);     
elseif (I==3)
    fprintf(fid,'C ');
fclose(fid);     
elseif (I==4)
    fprintf(fid,'D');
fclose(fid);     
elseif (I==5)
    fprintf(fid,'E');
fclose(fid);     
elseif (I==6)
    fprintf(fid,'F');
fclose(fid);     
elseif (I==7)
    fprintf(fid,'G');
fclose(fid);     
elseif (I==8)
    fprintf(fid,'H');
fclose(fid);     
elseif (I==9)
    fprintf(fid,'I');
fclose(fid);     
elseif (I==10)
    fprintf(fid,'J');
fclose(fid);     
elseif (I==11)
    fprintf(fid,'K');
fclose(fid);     
elseif (I==12)
    fprintf(fid,'L');
fclose(fid);     
elseif (I==13)
    fprintf(fid,'M');
fclose(fid);     
elseif (I==14)
    fprintf(fid,'N');
fclose(fid);     
elseif (I==15)
    fprintf(fid,'O');
fclose(fid);     
elseif (I==16)
    fprintf(fid,'P');
fclose(fid);     
elseif (I==17)
    fprintf(fid,'Q');
fclose(fid);     
elseif (I==18)
    fprintf(fid,'R');
fclose(fid);     
elseif (I==19)
    fprintf(fid,'S');
fclose(fid);     
elseif (I==20)
    fprintf(fid,'T');
fclose(fid);     
elseif (I==21)
    fprintf(fid,'U');
fclose(fid);     
elseif (I==22)
    fprintf(fid,'V');
fclose(fid);     
elseif (I==23)
    fprintf(fid,'W');
fclose(fid);     
elseif (I==24)
    fprintf(fid,'X');
fclose(fid);
elseif (I==25)
    fprintf(fid,'Y');
fclose(fid);     
elseif (I==26)
    fprintf(fid,'Z');
fclose(fid);     
 
 else
     disp(' not Found');
     
   clear
end
 
 
 