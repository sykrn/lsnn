function results = cvdata(x,y,cvprt,net)
%CVDATA Summary of this function goes here
%   Detailed explanation goes here       
    
    L= length(cvprt);
    
    results.trPerf = zeros(L,1);
    results.tsPerf = zeros(L,1);
    results.trtimePerf = zeros(L,1);
    results.tstimePerf = zeros(L,1);
    results.nNode = zeros(L,1);
    
    for i = 1:L
        idtr = cvprt(i).train;
        idts = cvprt(i).test;
        net = net.train(x(idtr),y(idtr));        
        results.trPerf(i) = mse(y(idtr),net.forward(x(idtr))); 
        results.trtimePerf(i) = net.traintime;
        
        t = cputime;
        yp = net.forward(x(idts));
        results.tstimePerf(i) = cputime - t;
        
        results.tsPerf(i) = mse(y(idts),yp);
        results.nNode(i) = size(net.weights{1},2);
    end
end

