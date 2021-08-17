% -------------------------------------------------------------------------
% GLOF demo
% Author: Gang Wang
% Email: gwang.cv@gmail.com
% -------------------------------------------------------------------------
% Requirement: 
% VLFeat Toolbox, http://www.vlfeat.org/download.html
% -------------------------------------------------------------------------
% Input: X, Y and the ground-truth index
% Output: Precision, Recall, F1-score, and Runtime (in seconds)
% -------------------------------------------------------------------------

data=load('sample.mat');
label=data.CorrectIndex;
X=data.X(:,1:2);
Y=data.Y(:,1:2);
GLOF(X,Y,label);