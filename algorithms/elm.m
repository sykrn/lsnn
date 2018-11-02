classdef elm
    %ELM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        cReg = 1e-6
        nHidden = 100
        weights = {}
        traintime = 0
        err = 0
    end
    
    methods
        function obj = elm(varargin)
            %ELM Construct an instance of this class
            %   Detailed explanation goes here
           if nargin > 0
               obj.nHidden=varargin{1};
               obj.cReg=varargin{2};
           end
        end
        
        function y = forward(obj,x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            n = size(x,1);
            y = tanh([ones(n,1) x]*obj.weights{1})*obj.weights{2};
        end
        
        function obj = train(obj,x,t)
            C=obj.cReg;
            N=obj.nHidden;
            
            start_time=cputime;
            [nInstances,nfeatures]=size(x);
            x=[ones(nInstances,1), x];
            w1=2*rand(nfeatures+1,N)-1;
            H=tanh(x*w1);

            clear x;
            w2=pinv(H'*H + eye(N)*C)*H'*t;
            obj.err = mse(t,H*w2);
            clear t;
            obj.traintime=cputime-start_time;
           

            %% construct the net
      
            obj.weights{1}=w1;
            obj.weights{2}=w2;
        end
    
    end
end

