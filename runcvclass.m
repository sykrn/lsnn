%% Classification datasets
clear
rng(0);
addpath(genpath(pwd()));

dpath ='D:\Codeplace\Dataset\GunarDataset\benchmarks.mat';
datalist = load(dpath,'benchmarks');
datalist = datalist.('benchmarks');
datanames = datalist;

L = length(datalist);

netnames = {'cpelm','lsm','elm','ielm','eielm','pcaelm','dpelm','lsm','ail','bpnet'};


ELMnode = [50    20    30    10    40    30   200    40   165    35    20    65    85];
BPnode = [15    45    45    10    75    10    15    50    80    25    25     5    10];
LSMiter=[8     2     2     4     1     3    10    10     7     7     2     1     3];

perfs=struct;

for idx = 1:1%length(netnames)  
    alname = netnames{idx}
    
    for k = 1:L        
        switch alname
            case 'lsm'
                net = lsm(LSMiter(k),0); %args: (iteration, isReg?)
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
                net = ail(BPnode(k)*2,1e-6); %args: (iteration, cReg/Lambda)
            otherwise
                net = bpnet(BPnode(k)); %args: (nHidden)
        end
        
        disp(datanames{k});
        dt=load(dpath,datalist{k});
        x=dt.(datalist{k}).x;
        y=dt.(datalist{k}).t;
        % to ensure the splits are the same for all algorithms
        cv=struct;
        for ii = 1:size(dt.(datalist{k}).test,1)
            cv(ii).test = dt.(datalist{k}).test(ii,:);
            cv(ii).training = dt.(datalist{k}).train(ii,:);            
        end        
        perfs.(alname).(datanames{k}) = runcvdata(x,y,cv,net,1);        
    end
    
end

save('performclasscpelm','perfs');



