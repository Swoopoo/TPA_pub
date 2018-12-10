function g = PLW(param,cmats,S,a_lw,iter)
%==========================================================================
%==========================================================================
%-----------------------|Bildrekonstruktion mit PLW|-----------------------
%==========================================================================
%==========================================================================
% Rekonstruktionalgorithmus 
%
% -INPUT-
%        S = Sensitvitätsmatrix
%        a_lw = Schrittweite
%        iter = Anzahl der Iterationen
%        param = 
%
%                 work_path: Arbeitspfad (char)
%           param.nlist_nam: Name der Knotendatei
%          param.elist_name: Name der Elementdatei
%           ansys_cmats_dir: Ordnername der Ansys-Kapazitätsmatrizen (char)
%                    er_max: Maximale Permittivität (double)
%                    er_min: Minimale Permittivität (double)
%                         m: Anzahl der Elektroden (int)
%                         N: Anzahl der Pixel (int)
%                         M: Anzahl der unab. Elektrodenpaarungen (int)
%                      norm: Art der Normierung. 's' oder 'p' (char)
%                      anim: Animationsoption. 1: Mit 0: Ohne (bin)
%                       ref: Referenzkap. der Messkarte PicoCap (double)
%                     elist: Koinzidenzliste der Pixel (N x 3 - int-Array)
%                     nlist: Knotenkoordinaten (? x 3 - double-Array)
%
%        cmats = 
%
%                    C_max: Ansys-Kapazitätsmatrix bei er_max (m x m)
%                    C_min: Ansys-Kapazitätsmatrix bei er_min (m x m)
%                      C_n: Cell mit N Ansys-Kapazitätsmatrizen (1 x N)
%                  C_m_max: Kapazitätsvektor bei er_max (M x 1)
%                  C_m_min: Kapazitätsvektor bei er_min (M x 1)
%              C_m_phantom: Kapazitätsvektor mit Phantom (M x 1)
% -OUTPUT-
%        g = Intensitätsvektor
%
%   Author(s): A. Bogner, M. Mösch
%   Copyright 2016 Chair of Measurement and Control Engineering,
%                  University of Bayreuth
%   $Revision: 1.0 $  $Date: 2016/06/21 $
%-------------------------------------------------------------------------- 
norm = param.norm; 
anim = param.anim;
C_phantom = cmats.C_phantom; 
%==========================================================================


%==========================================================================
%--------------------------Normierung der Kapazitäten----------------------
%==========================================================================
C = capNorm(param,cmats.C_m_min,cmats.C_m_max,C_phantom,norm);

%
%==========================================================================

%==========================================================================
%----------------Proj. Landweber Iterationsverfahren-----------------------
%==========================================================================
%
%Projektion:
%                  f(x)= g=g+a_lw*S'*(C'-S*g);
%                     _
%                    |   0       falls   f(x) < 0
%                    |
%         P[f(x)] = <   f(x)     falls  0 <= f(x) <= 1  
%                    |
%                    |_  1       falls f(x) > 1
%                    
%Startwert über LBP
g=S'*C;
N=size(g,1);
%g=SVD(param,cmats,S);
%
Z=zeros(1,size(param.nlist,1));
%Iteration
if anim == 1
    for ii=1:iter 
        g=g+a_lw*S'*(C-S*g);
        for jj=1:N
            if g(jj)<0 
                g(jj)=0;
            elseif g(jj)>=1
                g(jj)=1;
            else   
            end
        end
        trisurf(param.elist,param.nlist(:,1),param.nlist(:,2),Z,g);
        view(0,90);
        axis equal;
        drawnow;
     end
elseif anim == 0
    for ii=1:iter 
        g=g+a_lw*S'*(C-S*g);
        for jj=1:N
            if g(jj)<0 
                g(jj)=0;
            elseif g(jj)>=1
                g(jj)=1;
            else 
            end
        end
     end    
end
%  
%==========================================================================
