
function results = runcvdata(x,y,cvprt,net,class)
%CVDATA Summary of this function goes here
%   Detailed explanation goes here       
    
    L = length(cvprt);
    
    results.trPerf = zeros(L,1);
    results.tsPerf = zeros(L,1);
    results.trtimePerf = zeros(L,1);
    results.tstimePerf = zeros(L,1);
    results.nNode = zeros(L,1);
    
    
    for i = 1:L
        idtr = cvprt(i).training;
        idts = cvprt(i).test;
        net = net.train(x(idtr,:),y(idtr,:));          
        results.trtimePerf(i) = net.traintime;
        
        t = cputime;
        yp = net.forward(x(idts,:));
        results.tstimePerf(i) = cputime - t;
        
        if class            
            results.trPerf(i) = sum(y(idtr,:).*net.forward(x(idtr,:))<=0)/length(idtr);
            results.tsPerf(i) = sum(y(idts,:).*yp<=0)/length(idts);
        else
            results.trPerf(i) = mse(y(idtr,:),net.forward(x(idtr,:)));
            results.tsPerf(i) = mse(y(idts,:),yp);
        end
        
        results.nNode(i) = size(net.weights{1},2);
    end
end

