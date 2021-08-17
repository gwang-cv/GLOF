% -------------------------------------------------------------------------
% GLOF demo
% Author: Gang Wang
% Email: gwang.cv@gmail.com
% -------------------------------------------------------------------------

function GLOF(X,Y,label) 

% install VLFeat 
% cd vlfeat-0.9.21/toolbox/
% vl_setup
% cd ../../

beta1 =0.75;
testK = [10, 15, 20, 25, 30];
X(:,3)=1;
Y(:,3)=1;
[Xn, T] = normalise2dpts(X');
[Yn, T] = normalise2dpts(Y');
X=Xn(1:2,:);
Y=Yn(1:2,:);
X=double(X');
Y=double(Y');
 
%% Main procedure
Xt = X';
Yt = Y';
vec=Yt-Xt;
vx = vec(1, :);
vy = vec(2, :);
d2=vec(1,:).^2+vec(2,:).^2;
vec=vec';
tic;
kdtreeX = vl_kdtreebuild(Xt);
kdtreeY = vl_kdtreebuild(Yt);

[P,result] = LOF_MultiScaleScoreV2(vec,testK);
thresh=beta1 ; 
P= result < thresh.*ones(1,size(X,1));
idt = find(P == 1);
times=toc;
[TP,FP,TN,FN, precision, recall, accuracy, corrRate,F1] = evaluatePR_release(label, idt, size(X,1));
disp(['Precision=' num2str(precision) ', Recall=' num2str(recall) ', F1-score=' num2str(F1) ', Runtime = ' num2str(times) 's' ]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Functions    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Multi-Scale Score v2
    function [P,C] = LOF_MultiScaleScoreV2(A,testK)
        C=zeros(1,size(A,1));
        num=0;
        for ks = testK 
            [neighborX, ~] = vl_kdtreequery(kdtreeX, Xt, Xt, 'NumNeighbors', ks+1) ;
            [neighborY, ~] = vl_kdtreequery(kdtreeY, Yt, Yt, 'NumNeighbors', ks+1) ;
            [lofs,c] = LOFv2(A, ks,neighborX,neighborY);
            C=C+c;
            num=num+1;
        end
        C=(C)/num;
        P= C <= beta1.*ones(1,size(A,1));
    end

%% LOFv2
    function [lof,C] = LOFv2(A, k,neighborX,neighborY)
        if k < 1
            [numrows ~] = size(A);
            k = round(k*numrows);
        end
        
        [k_index, k_dist] = knnsearch(A,A,'k',k+1,'nsmethod','kdtree','IncludeTies',true);
        
        k_index = cellfun(@(x) x(2:end),k_index,'UniformOutput',false);
        numneigh = cellfun('length',k_index);
        
        k_dist1 = cell2mat(cellfun(@(x) x(end),k_dist,'UniformOutput',false));
        
        n = length(A(:,1));
        
        lrd_value = zeros(n,1);
        
        for i = 1:n 
            lrd_value(i) = lrd(A, i, k_dist1, k_index, numneigh(i));
        end
        
        lof = zeros(n,1);
        
        for i = 1:n
            lof(i) = sum(lrd_value(k_index{i})/lrd_value(i))/numneigh(i);
        end
     
        neighborX = neighborX(2:k+1, :);
        neighborY = neighborY(2:k+1, :);
        neighborIndex = [neighborX; neighborY];
        index = sort(neighborIndex);
        temp1 = diff(index);
        temp2 = (temp1 == zeros(size(temp1, 1), size(temp1, 2)));
        ni = sum(temp2); 
        d2i = d2(index);
        vxi = vx(index); vyi = vy(index);
        
        c1 = k-ni;   
        ratio = min(d2i, repmat(d2,size(d2i,1),1)) ./ max(d2i, repmat(d2,size(d2i,1),1));
        lof=1./(ones(size(lof,1),size(lof,2))+exp(1-lof)); 
        cos_sita= repmat(lof',size(ratio,1),1);
        alpha= 0.76; 
        c3i = cos_sita >= alpha.*ones(size(ratio, 1), size(ratio, 2));
        c3i0 = c3i(1:end-1, :).*temp2;  
        c3 = sum(c3i0); 
        C =  (c1 +  c3) / k;  
    end

%% lrd 
    function lrd_value = lrd(A, index_p, k_dist,k_index, numneighbors) 
        Temp = repmat(A(index_p,:), numneighbors, 1) - A(k_index{index_p}, :);
        Temp = sqrt(sum(Temp.^2,2)); 
        reach_dist = max([Temp k_dist(k_index{index_p})],[],2); 
        lrd_value = numneighbors/sum(reach_dist);
    end

%% evaluatePR_release
    function [TP,FP,TN,FN, precision, recall, accuracy, corrRate,F1] = evaluatePR_release(CorrectIndex, predictIndex, allpoints)
        % CorrectIndex is the inliers' index;
        if find(CorrectIndex>=2)  % if input inliers' index number such as 1,2,3,20,34...
            groundtruth= zeros(1,allpoints);
            groundtruth(CorrectIndex)=1;
            predict= zeros(1,allpoints);
            predict(predictIndex)=1;
        else   % if input index with only 0 and 1;
            groundtruth=CorrectIndex;
            predict=predictIndex;
        end
        TP = sum(groundtruth==1&predict==1);
        FP = sum(groundtruth-predict<0); 
        TN = sum(groundtruth==0&predict==0);
        FN = sum(groundtruth-predict>0); 
        
        corrRate = length(CorrectIndex)/allpoints;  
        precision = TP/(TP+FP);
        if (TP+FP)==0
            precision=0;
        end
        recall = TP/(TP+FN);
        if (TP+FN)==0
            recall=0;
        end
        accuracy=(TP+TN)/(TP+TN+FP+FN);
        if (TP+TN+FP+FN)==0
            accuracy=0;
        end
        F1=  2*(recall * precision) / (recall + precision);
        if  (recall + precision)==0
            F1=0;
        end
        
    end

end








