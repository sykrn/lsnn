classdef pcaelm
    %ELM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        cReg = 0
        confLev = 0.95
        weights = {}
        traintime = 0
        err = 0
    end
    
    methods
        function obj = pcaelm(varargin)
            %ELM Construct an instance of this class
            %   Detailed explanation goes here
           if nargin > 0
               obj.confLev=varargin{1};
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
            rng('shuffle');           
            
            start_time=cputime;
            [U,V,~]=svd(x'*x);
            V=diag(V);
            tot=sum(V)*obj.confLev;
            s=length(V);
            sv=0;
            n=1;
            for i=1:s
                sv=sv+V(i);
                if(sv >= tot)
                    n=i;
                    break;
                end
            end
            w1=U(:,1:n);
            clear U V;
            H=tanh(x*w1);
            clear x;
            w2=pinv(H'*H + eye(size(w1,2))*C)*H'*t;
            obj.err = mse(t,H*w2);
            clear t;

            obj.traintime=cputime-start_time;
           

            %% construct the net
      
            obj.weights{1}=w1;
            obj.weights{2}=w2;
        end
    
    end
end

