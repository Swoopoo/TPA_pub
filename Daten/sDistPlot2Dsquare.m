function sDistPlot2Dsquare(square,S,e_pair)
%SDISTPLOT2DSQUARE(param,S,e_paar,type) plots the sensitivity cards for 2D data.
% The following input is needed:
% - param.N
% - param.elist
% - param.nlist
% - S
% - e_pair
%
%   Author(s): M. Mösch
%   Copyright 2016 Chair of Measurement and Control Engineering,
%                  University of Bayreuth
%   $Revision: 1.0 $  $Date: 2016/06/21 $

Xi = square.x_vec;
Yi = square.y_vec;
[Xii,Yii] = meshgrid(Xi,Yi);

pp = 1;
for ii = 1:size(Yi,2)
    for jj = 1:size(Xi,2)
        Z_matrix(ii,jj) = S(e_pair,pp);
        pp = pp+1;
    end
end

surf(Xii,Yii,Z_matrix);
view(45,45);
axis square;
end