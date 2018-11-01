classdef eielm
    %ELM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        K = 1
        N = 100
        weights = {}
        traintime = 0
        err = 0
    end
    
    methods
        function obj = eielm(varargin)
            %ELM Construct an instance of this class
            %   Detailed explanation goes here
           if nargin > 0
               obj.N=varargin{1};
               obj.K=varargin{2};
           end
        end
        
        function y = forward(obj,x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            n = size(x,1);
            y = tanh([ones(n,1) x]*obj.weights{1})*obj.weights{2};
        end
        
        function obj = train(obj,x,t)
            k=obj.K;
            nHidden=obj.N;
            rng('shuffle');
            
            start_time=cputime;
            [nInstances,nfeatures]=size(x);  
            x=[ones(nInstances,1), x];
            w1=zeros(nfeatures+1,nHidden);    
            w2=zeros(nHidden,size(t,2));
            E=t;
            clear t;
            ev=E;
            eopt=trace(E'*E);


            for i=1:nHidden
                w1setk=2*rand(nfeatures+1,k)-1;        
                for j=1:k           
                    h=tanh(x*w1setk(:,j));       
                    w2_temp=h'*E/(h'*h);
                    e_temp=E-h*w2_temp;
                    en=trace(e_temp'*e_temp);
                    if(en<eopt)
                        w1(:,i)=w1setk(:,j);
                        w2(i,:)=w2_temp;
                        ev=e_temp;
                        eopt=en;
                    end 
                end
                E=ev;        
            end
 
            obj.traintime=cputime-start_time;
           

            %% construct the net
      
            obj.weights{1}=w1;
            obj.weights{2}=w2;
        end
    
    end
end

