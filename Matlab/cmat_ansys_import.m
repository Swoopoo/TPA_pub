function C = cmat_ansys_import(param,filename,e_num)
%C = CMAT_ANSYS_IMPORT gives back the C matrix from a file filename.
% The following input is needed:
% - filename (name of the c matrix file as string with .txt-ending)
% - param.M
% - param.m  : Anzahl der Elektroden, wird hier in e_num uebergeben
% - param.shuffle: 1: classical order
%                  2: Order is 12 23 34 ..., 13 24 35 .... 14 25 36..
%
%   Author(s): A. Bogner, M. Mösch
%   Copyright 2016 Chair of Measurement and Control Engineering,
%                  University of Bayreuth
%   $Revision: 1.0 $  $Date: 2016/06/21 $

delimiter = ' ';
startRow = 1;
endRow = inf;
formatSpec = '%30f';

for ii=1:e_num-2
    formatSpec = strcat(formatSpec,'%30f');
end

%-import c matrix----------------------------------------------------------

% formatSpec = strcat(formatSpec,'%f%[^\n\r]');
formatSpec = strcat(formatSpec,'%f');
fileID = fopen(filename,'r');
% dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
%for block=2:length(startRow)
 %   frewind(fileID);
 %   dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
 %   for col=1:length(dataArray)
%        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
%    end
%end
fclose(fileID);
C_work = [dataArray{1:end}];


%-converting to c vector---------------------------------------------------   
if param.shuffle == 1
    C = zeros(param.M,1);
    pp=1;
    for ii=1:(param.m-1)
        for jj=(ii+1):param.m         
            C(pp) = C_work(ii,jj);  
            pp=pp+1;
        end
    end 
elseif param.shuffle == 2
    i_end_in = param.m;
    i_end_out = floor((param.m-1)/2);
    i_c = 1;
    C = zeros(param.M,1);
    for i_out = 1:i_end_out
        for i_in = 1:i_end_in
            sec_ind = i_in+i_out;
            if sec_ind > param.m
                sec_ind = mod(sec_ind,param.m);
            end
            C(i_c) = C_work(i_in,sec_ind);
            i_c = i_c + 1;  
        end
    end
    if mod(param.m,2) == 0
        for i_in = 1:i_end_in/2
           sec_ind = i_in+param.m/2;
           C(i_c) = C_work(i_in,sec_ind);
           i_c = i_c + 1;
        end
    end
else
    printf('Wrong number in param.shuffle, must be 1 or 2');
    return

end
