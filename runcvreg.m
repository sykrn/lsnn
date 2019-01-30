%% Regression dataset

addpath(genpath(pwd()));

datapath = 'datasets\regbenchmark\';
datalist ={'abalone.csv','ailerons.csv','automgp.csv','bank.csv','boston.csv',...
    'california.csv','elevator.csv','servo.csv','cpu_small.data','machine.data','triazines.data','r_wpbc.data' };
datanames ={'abalone','ailerons','autompg','bank','boston',...
    'california','elevators','servo','compAct','machineCPU','triazines','breastCancer' };
n_testing=[2177,4129,200,3692,256,12640,5517,87,4192,109,86,94];
L = length(datalist);

netnames = {'lsm','elm','ielm','eielm','pcaelm','dpelm','cpelm','bpnet','ail'};



% hyperparameters
ELMnode = [25,45,30,190,50,80,125,30,125,10,10,10];
BPnode = [10,20,10,20,5,10,5,10,45,10,5,5];
LSMiter = [3,3,1,15,3,10,2,2,11,2,1,1];

iter = 50; % fifty trials

perfs=struct;

for idx = 1:length(netnames)  
    alname = netnames{idx}
    clear cv
    for k = 1:L        
        switch alname
            case 'lsm'
                net = lsmrf(LSMiter(k),1); %args: (iteration, isReg?)
            case 'elm'
                net = elm(ELMnode(k),1e-6); %args: (nHidden, cReg)
            case 'dpelm'
                net = dpelm(ELMnode(k)); %args: (nHidden)
            case 'cpelm'
                net = cpelm(ELMnode(k)); %args: (nHidden)
            case 'pcaelm'
                net = pcaelm(0.95, 1e-6); %args: (confLev,cReg)
            case 'eielm'
                net = eielm(ELMnode(k),10); %args: (nHidden)
            case 'ielm'
                net = eielm(ELMnode(k),1); %args: (nHidden)
            case 'ail'
                net = ail(BPnode(k)*2,1e-6); %args: (iteration,  cReg/Lambda)
            otherwise
                net = bpnet(BPnode(k)); %args: (nHidden)
        end
        disp(datanames{k});
        dataset=normalize(csvread([datapath,datalist{k}]));
        dataset(isnan(dataset))=0;
        x=dataset(:,1:(size(dataset,2)-1));
        y=mapminmax(dataset(:,size(dataset,2)),-1,1);
        
        % to ensure the splits are the same for all algorithms
        rng(0);
        cv(1) = cvpartition(y,'HoldOut',n_testing(k),'Stratify',false);
        for i = 1:iter-1
            cv(i+1) = repartition(cv(i));
        end
        
        perfs.(alname).(datanames{k}) = runcvdata(x,y,cv,net,0);
        
    end
    
end

save('performsregfinalpublish', 'perfs');



