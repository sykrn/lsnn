classdef bpnet
    %BPNET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nHidden
        model
        traintime
        weights
    end
    
    methods
        function obj = bpnet(N)
            %BPNET Construct an instance of this class
            %   Detailed explanation goes here
            obj.nHidden = N;
            net = fitnet(N);
            net.trainParam.showWindow=0;
            net.trainParam.epochs=100;
            net.divideParam.trainRatio = 0.85;
            net.divideParam.valRatio = 0.15;
            net.divideParam.testRatio = 0;
            net.performParam.normalization = 'None';
            obj.model =net;   
            obj.weights{1} = zeros(1,N);
        end
        
        function obj = train(obj,x,y)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            ts = cputime;
            obj.model = train(obj.model,x',y');
            obj.traintime = cputime -ts;
        end
        
        function yy = forward(obj,x)
            yy=obj.model(x')';
        end
    end
end

