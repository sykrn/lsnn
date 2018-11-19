% Hyperparameters searching for gunar datasets
addpath(genpath(pwd()));

% dpath ='D:\Codeplace\Dataset\GunarDataset\benchmarks.mat';
% datasets = load(dpath,'benchmarks');
% datasets = datasets.('benchmarks');

dpath = 'D:\Codeplace\Dataset\dataset_single\';
datasets ={'abalone.csv','ailerons.csv','automgp.csv','bank.csv','boston.csv',...
    'california.csv','elevator.csv','servo.csv','cpu_small.data','machine.data','triazines.data','r_wpbc.data' };
n_testing=[2177,4129,200,3692,256,12640,5517,87,4192,109,86,94];

% hyperparams
% nodes = 5:5:200;
nodes = 1:15;
ncv = 10;


bestnode = ones(1,length(datasets));

alname = 'lsm';
class = 0;


for j = 1:length(datasets)
    clear cv;
    prev_err = 100;
    fails = zeros(1,length(nodes));       
    for i=1:length(nodes)
%         [j i]
        switch alname
            case 'lsm'
                net = lsm(nodes(i),~class); %args: (iteration, isReg?)
            case 'elm'
                net = elm(nodes(i),1e-6); %args: (nHidden, cReg)
            case 'dpelm'
                net = dpelm(nodes(i)); %args: (nHidden)
            case 'cpelm'
                net = cpelm(nodes(i)); %args: (nHidden)
            case 'pcaelm'
                net = pcaelm(0.95, 1e-6); %args: (confLev,cReg)
            case 'eielm'
                net = eielm(nodes(i),10); %args: (nHidden)
            case 'ielm'
                net = eielm(nodes(i),1); %args: (nHidden)
            case 'ail'
                net = ail(nodes(i),0,1e-6); %args: (iteration, isClass?, cReg/Lambda)
            otherwise
                net = bpnet(nodes(i)); %args: (nHidden)
        end
        
        
        
        if class
            cv = struct;
            dt=load(dpath,datasets{j});
            for ii = 1:ncv
                cv(ii).test = dt.(datasets{j}).test(ii,:);
                cv(ii).training = dt.(datasets{j}).train(ii,:);            
            end
            x=dt.(datasets{j}).x;
            y=dt.(datasets{j}).t;
        else 
            dt=normalize(csvread([dpath,datasets{j}]));
            dt(isnan(dt))=0;
            x=dt(:,1:(size(dt,2)-1));
            y=dt(:,size(dt,2));
            cv(1) = cvpartition(y,'HoldOut',n_testing(j),'Stratify',false);
            for ii = 1:ncv-1
                cv(ii+1) = repartition(cv(ii));
            end
        end
        
        
        net.smParams(2)=10;
        net.smParams(1)=0.01;
        results = runcvdata(x,y,cv,net,class); 
               
        if mean(results.tsPerf)<prev_err
            prev_err = mean(results.tsPerf);
            bestnode(j) = nodes(i);
            fails(i)=0;
        else
            fails(i) = fails(i-1)+1; 
            if fails(i)>5
                break;
            end
        end
        
    end
    prev_err
    
end

save([alname,'2_nodesreg'],'bestnode');