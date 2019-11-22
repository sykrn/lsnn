clear

% datanames ={'abalone','ailerons','autompg','bank','boston',...
%     'california','elevators','servo','compAct','machineCPU','triazines','breastCancer' }';

% 
dpath ='D:\Codeplace\Dataset\GunarDataset\benchmarks.mat';
datalist = load(dpath,'benchmarks');
datalist = datalist.('benchmarks')';
datanames=datalist;

p=load('performclassv17.mat', 'perfs');
p =p.perfs;

L=length(datanames);

ELM = zeros(L,1);
CPELM = zeros(L,1);
DPELM = zeros(L,1);
AIL = zeros(L,1);
BP = zeros(L,1);
LSM = zeros(L,1);
EBELM = zeros(L,1);
% PCAELM = zeros(L,1);
EIELM = zeros(L,1);
IELM = zeros(L,1);

% choosing metrics
k='tsPerf'; %'tsPerf';%nNode %trtimePerf

f=@mean;
a=1i;
for i = 1:L
    LSM(i) = f(p.lsm.(datanames{i}).(k)) + std(p.lsm.(datanames{i}).(k))*a;
    ELM(i) = f(p.elm.(datanames{i}).(k))+ std(p.elm.(datanames{i}).(k))*a;
    CPELM(i) = f(p.cpelm.(datanames{i}).(k)) + std(p.cpelm.(datanames{i}).(k))*a;
    DPELM(i) = f(p.dpelm.(datanames{i}).(k)) + std(p.dpelm.(datanames{i}).(k))*a;
    AIL(i) = f(p.ail.(datanames{i}).(k))+std(p.ail.(datanames{i}).(k))*a;
    BP(i) = f(p.bpnet.(datanames{i}).(k))+std(p.bpnet.(datanames{i}).(k))*a ;
    EBELM(i) = f(p.ebelm.(datanames{i}).(k))+std(p.ebelm.(datanames{i}).(k))*a;
%     PCAELM(i) = f(p.pcaelm.(datanames{i}).(k))+std(p.pcaelm.(datanames{i}).(k))*a;
    EIELM(i) = f(p.eielm.(datanames{i}).(k))+std(p.eielm.(datanames{i}).(k))*a;
    IELM(i) = f(p.ielm.(datanames{i}).(k))+std(p.ielm.(datanames{i}).(k))*a; 
end

ff = @(x)(round(x,4));
LSM = ff(LSM);
ELM= ff(ELM);
CPELM = ff(CPELM);
DPELM = ff(DPELM) ;
AIL = ff(AIL);
BP= ff(BP);
EBELM = ff(EBELM);
EIELM = ff(EIELM);
IELM = ff(IELM);
% PCAELM = ff(PCAELM);

t= table(datanames,LSM,AIL,ELM,IELM,EIELM,EBELM,DPELM,CPELM,BP)
writetable(t,'performclassv17')
