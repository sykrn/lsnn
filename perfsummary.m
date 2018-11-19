clear

% datanames ={'abalone','ailerons','autompg','bank','boston',...
%     'california','elevators','servo','compAct','machineCPU','triazines','breastCancer' }';


dpath ='D:\Codeplace\Dataset\GunarDataset\benchmarks.mat';
datalist = load(dpath,'benchmarks');
datalist = datalist.('benchmarks')';
datanames=datalist;

p=load('performclasslsm.mat', 'perfs');
p =p.perfs;

L=length(datanames);

ELM = zeros(L,1);
CPELM = zeros(L,1);
DPELM = zeros(L,1);
AIL = zeros(L,1);
BP = zeros(L,1);
LSM = zeros(L,1);
PCAELM = zeros(L,1);
EIELM = zeros(L,1);
IELM = zeros(L,1);
k='trtimePerf';

f=@mean;
% f=@std;
a=1i;
for i = 1:L
    LSM(i) = f(p.lsm.(datanames{i}).(k)) + std(p.lsm.(datanames{i}).(k))*a;
%     ELM(i) = f(p.elm.(datanames{i}).(k))+ std(p.elm.(datanames{i}).(k))*a;
%     CPELM(i) = f(p.cpelm.(datanames{i}).(k)) + std(p.cpelm.(datanames{i}).(k))*a;
%     DPELM(i) = f(p.dpelm.(datanames{i}).(k)) + std(p.dpelm.(datanames{i}).(k))*a;
%     AIL(i) = f(p.ail.(datanames{i}).(k))+std(p.ail.(datanames{i}).(k))*a;
%     BP(i) = f(p.bpnet.(datanames{i}).(k))+std(p.bpnet.(datanames{i}).(k))*a ;
%     PCAELM(i) = f(p.pcaelm.(datanames{i}).(k))+std(p.pcaelm.(datanames{i}).(k))*a;
%     EIELM(i) = f(p.eielm.(datanames{i}).(k))+std(p.eielm.(datanames{i}).(k))*a;
%     IELM(i) = f(p.ielm.(datanames{i}).(k))+std(p.ielm.(datanames{i}).(k))*a; 
end

ff = @(x)(round(x,3));

LSM = ff(LSM);
% ELM= ff(ELM);
% CPELM = ff(CPELM);
% DPELM = ff(DPELM) ;
% AIL = ff(AIL);
% BP= ff(BP);
% PCAELM = ff(PCAELM);
% EIELM = ff(EIELM);
% IELM = ff(IELM);

% table(datanames,LSM,AIL,ELM,IELM,EIELM,PCAELM,DPELM,CPELM,BP)

table(datanames,LSM)
