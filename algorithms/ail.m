    classdef ail
    %AIL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        maxNodes = 50
        lambdaLS = 1e-6
        weights = {}
        traintime = 0
        fwtime = 0
        err = 0
        useNorm = 0
        eta = 1e-4
    end
    
    methods
        function obj = ail(mn,lambda)
            %AIL Construct an instance of this class
            %   Detailed explanation goes here
            obj.maxNodes = mn;
            obj.lambdaLS = lambda;
        end
        
       
        function y = forward(obj,x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            n = size(x,1);
            y = [ones(n,1) tanh([ones(n,1) x]*obj.weights{1})]*obj.weights{2};
            
        end
        
        function obj = train(obj,x,t)
            starting_time=cputime;
            
            %% Initiation of learning
            % - read the number of instances and features
            % - add bias node / vector ones to the input features
            n_instances=size(x,1);            
            n_features=size(x,2);
            ipt=[ones(n_instances,1) x];
            % calculate the inverse of input matrix
            pinvIpt = pinv(ipt'*ipt+eye(n_features+1)*obj.lambdaLS)*ipt';
            
            % initiate the error vector using max value of er, E(er), for multioutput case
            er1=sum(t.^2,1);
            [~,id] = max(er1);
            er = t(:,id);
            
            % initiate and allocate the weights
            N=obj.maxNodes;
            w1=zeros(n_features+1,N);
            w2=[];
            
            % initiate and allocate hidden matrix H and set up the bias node/
            % vector one
            H=zeros(n_instances,N+1);
            H(:,1)=ones(1,n_instances);
            
            % this is for current s=pinv(H'H)
            inv_s=zeros(N+1,N+1);
            inv_s(1,1)=1/n_instances;
            
            % this is for u=H'y
            u=zeros(N+1,size(t,2));
            u(1,:)=H(:,1)'*t;
            
            ulbound=0.99;
            
            % initiate current rmse
            rmse=1000;
            for i=1:N
                rmse_prev=rmse;
                                
                if (obj.useNorm)
                    % normalize using z distribution and clipping on
                    % ulbound
                    er = normcdf(er); % 98.75% of distribution
                    idx = er(abs(er)>ulbound);
                    er(idx) = ulbound.*sign(er(idx));              
%                     er=atanh(er);                
                else
                    % normalize to +/- tanh(1) or [-0.99,0.99] as tanh function range [-1,1]
                    er=er-mean(er);
                    mmax=max(er(:));mmin=min(er(:));
                    m=max(abs(mmax),abs(mmin));
                    er=atanh((ulbound/m)* er);
                end
                
                % calcaulate and extend the input weight as increasing hidden node
                
                w1(:,i)=pinvIpt*er;
                H(:,i+1)=tanh(ipt*w1(:,i));
                
                % recursively compute hidden node weight matrix
                v=H(:,1:i)'*H(:,i+1);
                d=H(:,i+1)'*H(:,i+1);
                theta=inv_s(1:i,1:i)*v;
                alpha=d-v'*theta;
                
                % end the training or stop adding new hidden node when independend propertise (idp) is small enough
                if (abs(alpha/d)<=obj.eta)
                    w1(:,i:N)=[];
                    break;
                end
                
                inv_s(i+1,i+1)=1/alpha;
                inv_s(1:i,i+1)=-theta*inv_s(i+1,i+1);
                inv_s(i+1,1:i)=inv_s(1:i,i+1)';
                inv_s(1:i,1:i)=inv_s(1:i,1:i)+inv_s(i+1,i+1)*(theta*theta');
                
                u(i+1,:)=H(:,i+1)'*t;
                w2=inv_s(1:(i+1),1:(i+1))*u(1:i+1,:);
                
                er=t-H(:,1:(i+1))*w2;
                
                rmse=sqrt(mse(er));
                
                % end the training or stop adding new hidden node when relative error is small enough
                erelative=abs((rmse_prev-rmse)/rmse);
                if erelative<=obj.eta
                    w1(:,(i+1):N)=[];
                    break;
                end
                %         er1=sum(er,2);
                er1=sum(er.^2,1);
                [~,id] = max(er1);
                er = er(:,id);
            end
            
            stopping_time=cputime;
            trtime=stopping_time-starting_time;
            clear inv_s;
            ww={w1,w2};
            obj.err=rmse;
            obj.weights=ww;
            obj.traintime = trtime;
            
        end
    end
end

