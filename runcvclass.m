%% Classification datasets
clear
rng(0);
addpath(genpath(pwd()));

dpath ='datasets\gunarbenchmark\benchmarks.mat';
datalist = load(dpath,'benchmarks');
datalist = datalist.('benchmarks');
datanames = datalist;

L = length(datalist);

netnames = {'lsm','elm','ielm','eielm','ebelm','pcaelm','dpelm','cpelm','bpnet','ail'};

% hyperparameters
ELMnode = [50    20    30    10    40    30   200    40   165    35    20    65    85];
BPnode = [15    45    45    10    75    10    15    50    80    25    25     5    10];
LSMiter=[11,1,1,3,4,2,15,15,7,6,4,1,5]; %[11,1,1,4,4,2,15,15,4,8,4,1,7];%[6,1,1,5,1,2,15,15,7,7,5,1,4];

perfs=struct;

for idx = 1:length(netnames)  
    alname = netnames{idx}
    
    for k = 1:L        
        switch alname
            case 'lsm'
                net = lsmr(LSMiter(k),0); %args: (iteration, isReg?)
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
            case 'ebelm'
                net = ebelm(ELMnode(k),10); %args: (nHidden)
            case 'ielm'
                net = eielm(ELMnode(k),1); %args: (nHidden)
            case 'ail'
                net = ail(BPnode(k)*2,1e-6); %args: (iteration, cReg/Lambda)
            otherwise
                net = bpnet(BPnode(k)); %args: (nHidden)
        end
        
        disp(datanames{k});
        dt=load(dpath,datalist{k});
        x=dt.(datalist{k}).x; 
        y=dt.(datalist{k}).t;      
        cv=struct;
        for ii = 1:size(dt.(datalist{k}).test,1)
            cv(ii).test = dt.(datalist{k}).test(ii,:);
            cv(ii).training = dt.(datalist{k}).train(ii,:);            
        end        
        perfs.(alname).(datanames{k}) = runcvdata(x,y,cv,net,1);        
    end    
end

save('performclassv17','perfs');



