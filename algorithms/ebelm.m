classdef ebelm
    %ELM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        K = 1
        N = 100
        weights = {}
        traintime = 0
        err = 0
        eta =1e-4
        params = {}
    end
    
    methods
        function obj = ebelm(varargin)
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
            y = tanh([ones(n,1) x]*obj.weights{1});
            nNode = size(y,2);
            for i = 2:2:nNode 
                y(:,i) = mapminmax('reverse',y(:,i)',obj.params{i/2})';   
            end
            y = y * obj.weights{2};
        end
        
        function obj = train(obj,x,t)
            k=obj.K;
            nHidden=obj.N;
            
            start_time=cputime;
            [nInstances,nfeatures]=size(x);  
            pinvx = pinv(x);
            
            x=[ones(nInstances,1), x];
           
            w1=zeros(nfeatures+1,nHidden);    
            w2=zeros(nHidden,size(t,2));
            
            E=t;
            clear t;
            ev=E;
            eopt=trace(E'*E);


            for i=1:nHidden
                
                if (mod(i,2) == 1)
                    w1setk=2*rand(nfeatures+1,k)-1;
                    w1setk(1) = (w1setk(1)+1)/2;
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
                else
                    h2n=E./w2(i-1,:);
                    [h2n, par] = mapminmax(h2n',-0.9,0.9);
                    obj.params{i/2} = par;
                    
                    h2n = atanh(h2n');
                    
                    w1(2:nfeatures+1,i) = pinvx * h2n;
                    h = x(:,2:nfeatures+1)*w1(2:nfeatures+1,i);
                    w1(1,i) = sqrt(mse(h2n - h));
                    
                    h = mapminmax('reverse',tanh(h + w1(1,i))',par)';                    
                    w2(i,:)=h'*E/(h'*h);                   
                    ev=E-h*w2(i,:);  
                    
                end
                E=ev;
                if (mean(E.*E) < obj.eta)
                    w1 = w1(:,1:i);
                    w2 = w2(1:i,:);
                    break
                end
            end
 
            obj.traintime=cputime-start_time;
           

            %% construct the net
      
            obj.weights{1}=w1;
            obj.weights{2}=w2;
        end
    
    end
end

