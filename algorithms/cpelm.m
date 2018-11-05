classdef cpelm
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
        function obj = cpelm(varargin)
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
            clear x;
            [T,R]=qr(H,t,0);    
            clear H t;

            fnormT=trace(T'*T); 
            rIdx=1:N;
            Lfinal=0;

            for i=1:N
                maxIdx=i;
                maxEr=1e10;
                for j=i:N                      
                    [tt,~]=qr(R(i:N,j),T(i:N,:),0);
                    tSqNorm=tt*tt';
                    if(tSqNorm>maxEr)               
                        maxEr=tSqNorm;
                        maxIdx=j;                          
                    end            
                end

                % swap column
                rIdx(:,[i,maxIdx])=rIdx(:,[maxIdx,i]);
                R(:,[i,maxIdx])=R(:,[maxIdx,i]);

                [Temp,R(i:N,i)]=qr(R(i:N,i),[R(i:N,i+1:N) T(i:N,:)]);
                R(i:N,i+1:N)=Temp(:,1:N-i);
                Temp(:,1:N-i)=[];
                T(i:N,:)=Temp;
                clear Temp;

                if (trace(T(1:i,:)'*T(1:i,:))>=fnormT*(1-eps))
                    Lfinal=i;
                    break;
                end        
            end

            w1=w1(:,rIdx(1:Lfinal));
            w2=pinv(full(R(1:Lfinal,1:Lfinal)))*T(1:Lfinal,:);    
            obj.traintime=cputime-start_time;
            clear R T;

            %% construct the net      
            obj.weights{1}=w1;
            obj.weights{2}=w2;
        end
    
    end
end

