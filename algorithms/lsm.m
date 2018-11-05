classdef lsm
    %LSM (Local sigmoid method) Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        iter = 1
        reg = 1
        smParams = [0.02,10]
        weights = {}
        traintime = 0
        err = 0
        lambda = 1e-6
    end
    
    methods
        function obj = lsm(iter,reg)
            %LSM Construct an instance of this class
            %   Detailed explanation goes here
            obj.iter = iter;
            obj.reg = reg;
        end
        
        function y = forward(obj,x)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            n = size(x,1);
            y = [ones(n,1) tanh([ones(n,1) x]*obj.weights{1})]*obj.weights{2};
        end
        
        function obj = train(obj,x,t)
            starting_time=cputime;
            
            [n,m]=size(x);
            xx = [ones(n,1) x];
            y=t;
            
            % regularize
            C = diag([0 ones(1,m)]*obj.lambda);
            invxx = xx'*xx + C;
            span = obj.smParams(1)*n;
            sh = floor(span/2);
            span = 2*sh+1;
            if span < 3
                span = 3;
            end
            w=[];
            
            for i = 1:obj.iter
                w0 = invxx\(xx'*y);
                yp = xx*w0;
                
                %sort by prediction axis
                [~,ii] = sort(yp);
                xx=xx(ii,:);
                yp=yp(ii,:);
                t=t(ii,:);
                y=y(ii,:)-yp;
                
                %smooth iteratively using moving average
                for j=1:obj.smParams(2)
                    y = smooth(y,span);
                end
                
                % bend detection
                iid1=locopt(y,1);
                % preset thresholding
                if obj.reg
                    th = 1/4;
                else
                    th = 1;
                end
                
                % reduce points
                iid=clustpoint(iid1,y,th);
                lid = length(iid);
                if lid >0
                    % calculate each segments
                    for j=1:lid
                        idx = yp>yp(iid1(iid(j))) & yp < yp(iid1(iid(j)+1));
                        ww = fitLS(xx(idx,:),t(idx,:),obj.lambda);
                        if norm(ww(2:m+1))>1e-4
                            w=[w ww];
                        end
                    end
                end
                if isempty(w)
                    w=[w w0];
                end
                % second weight and error calculation
                
                hh = [ones(n,1) tanh(xx * w)];
                w2 = (hh'*hh + diag([0 ones(1,size(w,2))]*obj.lambda))\(hh'*t);
                y = t - hh*w2; % reduce the error iteratively
            end
            obj.err = mse(y);
            obj.traintime=cputime - starting_time;
            obj.weights={w,w2};            
        end
        
    end
end

function w = fitLS(x,y,C)
    %METHOD1 Summary of this method goes here
    %   Detailed explanation goes here
    s = 1;
    y = mapminmax(y',-s,s)';
    [n,m] =size(x);
     
    if n>=m        
        C = diag([0 ones(1,m-1)]*C);
        w=(x'*x+C)\(x'*y);
    else
        C = diag(ones(1,n)*C);
        w=x'*((x*x'+C)\y);
    end
end

function idx = locopt(y,step)
    l=length(y);
    lo = y(step+1:l-step)-y(1:l-2*step);
    hi = y(2*step+1:l)-y(step+1:l-step);

    lh=lo.*hi;

    yy=zeros(l,1);
    yy(1)=-1;
    yy(l)=-1;
    yy(step+1:l-step)=lh;

    idx = find(yy<0);
end

function yy = grad2(y,xspan)
    l = length(y);
    xstep = xspan/l;
    h2= xstep;
    yy = ([y(2:l);0] - (2*y) + [0;y(1:l-1)])/h2;
    yy([1,l])=0;
end

function iid = clustpoint(idx,x,s)
    l=length(idx);

    d=abs(x(idx(2:l))-x(idx(1:l-1)));
    m=mean(d);
    iid=find(d > m * s );

end

