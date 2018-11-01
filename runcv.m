%% Regression dataset

addpath(genpath(pwd()));

datapath = 'D:\Codeplace\Dataset\dataset_single\';
datalist ={'abalone.csv','ailerons.csv','automgp.csv','bank.csv','boston.csv','california.csv','elevator.csv','servo.csv'};
n_testing=[2177,4129,200,3692,256,12640,5517,87];
L = length(datalist);

iter = 50;
net = lsm(1,1);
% net =elm(25,0);

for k = 8:8
    dataset=mapminmax(csvread([datapath,datalist{k}])',-1,1)';
    x=dataset(:,1:(size(dataset,2)-1));
    y=dataset(:,size(dataset,2));
    
    cv(1) = cvpartition(y,'HoldOut',n_testing(k),'Stratify',false);
    
    for i = 1:iter-1
        cv(i+1) = repartition(cv(i));
    end
    
    results = runcvdata(x,y,cv,net);   
end
