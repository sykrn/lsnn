classdef dpelm
    %ELM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nHidden = 100
        weights = {}
        traintime = 0
        eta =1e-3
        err = 0
    end
    
    methods
        function obj = dpelm(varargin)
            %ELM Construct an instance of this class
            %   Detailed explanation goes here
           if nargin > 0
               obj.nHidden=varargin{1};
           end
        end
        
        function y = forward(obj,x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            n = size(x,1);
            y = tanh([ones(n,1) x]*obj.weights{1})*obj.weights{2};
        end
        
        function obj = train(obj,x,t)
            
            N=obj.nHidden;            
      
            eps=obj.eta;

            start_time=cputime;    
            [nInstances,nfeatures]=size(x);
            x=[ones(nInstances,1), x];
            w1=2*rand(nfeatures+1,N)-1;

            H=tanh(x*w1);
            H=sparse(H);
            [T,R]=qr(H,t,0);  

            clear H;
            fnormT=trace(T'*T); 
            rIdx=1:N;            

            for i=1:N
                minIdx=N-i+1;
                minEr=T(N-i+1,:)*T(N-i+1,:)';
                for j=1:(N-i)                      
                    [tt,~]=qr(R(j:N-i+1,j+1:N-i+1),T(j:N-i+1,:));            
                    tSqNorm=tt(N-i+2-j,:)*tt(N-i+2-j,:)';
                    if(tSqNorm<minEr)               
                        minEr=tSqNorm;
                        minIdx=j;                          
                    end  
                end
                % swap
                rIdx(:,minIdx)=[];
                R(:,minIdx)=[];

                if ~(minIdx==N-i+1)
                    [T(minIdx:N,:),R(minIdx:N,minIdx:(N-i))]=qr(R(minIdx:N,minIdx:(N-i)),T(minIdx:N,:));
                end

                if (trace(T(1:(N-i),:)'*T(1:(N-i),:))<=fnormT*(1-eps))                  
                    break;
                end     
            end

            w1=w1(:,rIdx);
            w2=R\T;    
            obj.traintime=cputime-start_time;
            clear R T;


            %% construct the net      
            obj.weights{1}=w1;
            obj.weights{2}=w2;            
        end
    
    end
end

