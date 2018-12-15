function C_norm = capNorm(param,C_min,C_max,C,norm)
%C_norm = CAPNORM(param,C_min,C_max,C,norm) normalises the vector C and 
%returns the normalised vector C_norm.
% The following input is needed:
% - param.m
% - param.M
% - C_min
% - C_max
% - C
% - norm ('s' for serial or 'p' for parallel norm)
%
%   Author(s): A. Bogner, A. Fischerauer
%   Copyright 2016 Chair of Measurement and Control Engineering,
%                  University of Bayreuth
%   $Revision: 1.0 $  $Date: 2016/06/21 $
%   $Revision: 1.1 $  $Date: 2017/04/21 $


if param.ind_norm == 1
    switch norm
        case 's'
            C_norm = zeros(param.M,1);
            for ii=1:param.M            
                C_norm(ii) = (1/C(ii)-1/C_min(ii))/(1/C_max(ii)-1/C_min(ii));
            end

        case 'p'
            C_norm = zeros(param.M,1);
            for ii=1:param.M
            C_norm(ii) = (C(ii)-C_min(ii))/(C_max(ii)-C_min(ii));
            end

        otherwise
            error('wrong norm');
    end 
elseif param.ind_norm == 2 && param.shuffle == 2
    norm_vec = param.norm_vec;
    row_num_sou = param.N/param.m;
    row_index_sou = floor((param.var-1)/row_num_sou)+1;
    C_norm = zeros(param.M,1);
    for ii=1:param.M
        if norm_vec(ii) == 's'
            C_norm(ii) = (1/C(ii)-1/C_min(ii,row_index_sou))...
                /(1/C_max(ii,row_index_sou)-1/C_min(ii,row_index_sou));
        elseif norm_vec(ii) == 'p'
            C_norm(ii) = (C(ii)-C_min(ii,row_index_sou))...
                /(C_max(ii,row_index_sou)-C_min(ii,row_index_sou));
        else
            error('wrong norm');
        end
    end
    
else
    error('wrong norm');
end
end



