% Ariel edit: found on https://github.com/zhuhufei/OMP/blob/master/codeAug2020.m

%Code for OMP implementations submitted to Electronics (ISSN 2079-9292)

%This article shares the MATLAB code to generate Fig. 1 - Fig. 6 and of the following paper: Hufei Zhu, Wen Chen, and Yanpeng Wu,
%"Efficient Implementations for Orthogonal Matching Pursuit", submitted to Electronics (ISSN 2079-9292) in 2020.

 

 

%References for what follows:

%[1] B.L. Sturm and M.G. Christensen,  http://www.eecs.qmul.ac.uk/~sturm/software/OMPefficiency.zip
%[2] B.L. Sturm and M.G. Christensen, ¡°Comparison of orthogonal matching pursuit implementations¡±, EUSIPCO 2012.


%I. Code for Fig. 1, Fig. 2 and Fig. 3 of submitted paper

%Please visit [1] to download the shared matlab code for the reference [2]. Then find the code to generate Fig.1 of [2]. 
%To obtain the simulation results for the proposed implementations, please replace the corresponding function with ForErrorProposedv0, 
%ForErrorProposedv1, ForErrorProposedv2, ForErrorProposedv3 or ForErrorProposedv4, which is as follows.

 


function [s,amplitudes,innerprod] = ForErrorProposedv0(x, dict, D, dictTx, natom, tolerance)
% Proposed OMP Implementation (v0).
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary (with unit norm columns)
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
innerprod{1}=projections(:);
k = 1;
D_mybest = zeros(size(D,1),natom);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
   [alpha,maxindex] = max(projections.^2);
   newgam = maxindex(1);
   gamma(k) = newgam;
   if k==1
      D_mybest(:,1) = D(:,newgam);
      a_F(1)= projections(newgam);
      projections = projections -D_mybest(:,1)*a_F(1);
      normr2 = normr2 -a_F(1)^2;
   else
      temp_F_k_k =sqrt(1/(1- sum(D_mybest(newgam,:).^2)));
      F(:,k)=-temp_F_k_k*(F*D_mybest(newgam,:).');
      F(k,k)=temp_F_k_k;
      D_mybest(:,k)=temp_F_k_k*(D(:,newgam)-D_mybest*D_mybest(newgam,:).');
      a_F(k)=temp_F_k_k*projections(newgam);
      projections = projections -D_mybest(:,k)*a_F(k);
      normr2 = normr2 -a_F(k)^2;
   end
  amplitudes{k}=zeros(dictsize,1);
  amplitudes{k}(gamma(1:k))=F(1:k,1:k)*a_F(1:k);
   k = k + 1;
  innerprod{k}=projections(:);
end
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));

 


function [s,amplitudes,innerprod] = ForErrorProposedv1(x, dict, D, dictTx, natom, tolerance)
% Proposed Memory-Saving version 1.
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary (with unit norm columns)
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
innerprod{1}=projections(:);
k = 1;
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
   [alpha,maxindex] = max(projections.^2);
   newgam = maxindex(1);
   gamma(k) = newgam;
   if k==1
      a_F(1)= projections(newgam);
      projections = projections -D(:,newgam)*a_F(1);
      normr2 = normr2 -a_F(1)^2;
   else
      s_k_minus_1 = D(gamma(1:k-1),newgam);
      c_k_minus_1 = F(1:k-1,1:k-1)'*s_k_minus_1;
      temp_F_k_k =sqrt(1/(1- sum(c_k_minus_1.^2)));
      F(1:k-1,k)=-temp_F_k_k*(F(1:k-1,1:k-1)*c_k_minus_1);
      F(k,k)=temp_F_k_k;
      a_F(k)=temp_F_k_k*projections(newgam);
      projections = projections -D(:,gamma(1:k))*(F(1:k,k)*a_F(k));
      normr2 = normr2 -a_F(k)^2;
   end
  amplitudes{k}=zeros(dictsize,1);
  amplitudes{k}(gamma(1:k))=F(1:k,1:k)*a_F(1:k);
   k = k + 1;
  innerprod{k}=projections(:);
end
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));


function [s,amplitudes,innerprod] = ForErrorProposedv2(x, dict, D, dictTx, natom, tolerance)
% Proposed Memory-Saving version 2.
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary (with unit norm columns)
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%

x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
innerprod{1}=projections(:);
k = 1;
D_mybest = zeros(dictsize,natom);
temp_D_row = zeros(natom,1);
E = zeros(size(dict,1),natom);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   D_mybest(:,1) = dict.'*dict(:,newgam);
   E(:,1)=dict(:,newgam);
   a_F(1)= projections(newgam);
  projections = projections -D_mybest(:,1)*a_F(1);
   normr2 = normr2 -a_F(1)^2;
 else
temp_D_row = D_mybest(newgam,:).';
temp_F_k_k =sqrt(1/(1- sum(temp_D_row.^2)));
F(:,k)=-temp_F_k_k*(F*temp_D_row);
F(k,k)=temp_F_k_k;
 E(:,k)=temp_F_k_k*(dict(:,newgam)-E*temp_D_row);
 D_mybest(:,k)=dict.'*E(:,k);
   a_F(k)=temp_F_k_k*projections(newgam);
projections = projections -D_mybest(:,k)*a_F(k);
  normr2 = normr2 -a_F(k)^2;
 end
   amplitudes{k}=zeros(dictsize,1);
  amplitudes{k}(gamma(1:k))=F(1:k,1:k)*a_F(1:k);
 k = k + 1;
  innerprod{k}=projections(:);
end
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));

 

 function [s,amplitudes,innerprod] = ForErrorProposedv3(x, dict, D, dictTx, natom, tolerance)
% Proposed Memory-Saving version 3.
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary (with unit norm columns)
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%

x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;

F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
innerprod{1}=projections(:);
k = 1;
temp_D_row = zeros(natom,1);
E = zeros(size(dict,1),natom);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   E(:,1)=dict(:,newgam);
   a_F(1)= projections(newgam);
projections = projections -dict.'*(E(:,1)*a_F(1));
   normr2 = normr2 -a_F(1)^2;
 else
temp_D_row = (dict(:,newgam).'*E).';
temp_F_k_k =sqrt(1/(1- sum(temp_D_row.^2)));
F(:,k)=-temp_F_k_k*(F*temp_D_row);
F(k,k)=temp_F_k_k;
 E(:,k)=temp_F_k_k*(dict(:,newgam)-E*temp_D_row);
   a_F(k)=temp_F_k_k*projections(newgam);
projections = projections -dict.'*(E(:,k)*a_F(k));
  normr2 = normr2 -a_F(k)^2;
 end
   amplitudes{k}=zeros(dictsize,1);
  amplitudes{k}(gamma(1:k))=F(1:k,1:k)*a_F(1:k);
 k = k + 1;
  innerprod{k}=projections(:);
end
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));

 
 function [s,amplitudes,innerprod] = ForErrorProposedv4(x, dict, D, dictTx, natom, tolerance)
% Proposed Memory-Saving version 4.
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary (with unit norm columns)
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%

x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;

F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
innerprod{1}=projections(:);
k = 1;
temp_D_row = zeros(natom,1);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   a_F(1)= projections(newgam);
projections = projections -dict.'*(dict(:,gamma)*(F(1,1)*a_F(1)));
   normr2 = normr2 -a_F(1)^2;
 else   
g_lamda_k = dict(:,newgam).'*dict(:,newgam);
s_k_minus_1 = dict(:,gamma(1:k-1)).'*dict(:,newgam);
c_k_minus_1 = F(1:k-1,1:k-1)'*s_k_minus_1;
temp_F_k_k =sqrt(1/(1- sum(c_k_minus_1.^2)));
F(:,k)=-temp_F_k_k*(F*c_k_minus_1);
F(k,k)=temp_F_k_k;
   a_F(k)=temp_F_k_k*projections(newgam);
projections = projections -dict.'*(dict(:,gamma)*(F(1:k,k)*a_F(k)));
  normr2 = normr2 -a_F(k)^2;
 end
   amplitudes{k}=zeros(dictsize,1);
  amplitudes{k}(gamma(1:k))=F(1:k,1:k)*a_F(1:k);
 k = k + 1;
  innerprod{k}=projections(:);
end
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));
 

 

 

% II. Code for Fig 4,  Fig. 5 and Fig. 6 of submitted paper

%Please visit [1] to download the shared matlab code for the reference [2]. Then find the code to generate Fig.2 of [2].
%To obtain the simulation results for the proposed implementations, please replace the corresponding function with ForTimeProposedv0, ForTimeProposedv1, ForTimeProposedv2, 
%ForTimeProposedv3 or ForTimeProposedv4, which is as follows.


function [s,tout] = ForTimeProposedv0(x, dict, D, dictTx, natom, tolerance)
clear dict;
% Actually dict, i.e., the dictionary, is not required.
% Proposed OMP Implementation (v0).
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize =  size(D,2); %size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
tout = zeros(natom,1);
solveropts.UT = true;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
k = 1;
solveropts.UT = true;
D_mybest = zeros(size(D,1),natom);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 tstart = tic;
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   D_mybest(:,1) = D(:,newgam);
   a_F(1)= projections(newgam);
  projections = projections -D_mybest(:,1)*a_F(1);
   normr2 = normr2 -a_F(1)^2;
 else
temp_F_k_k =sqrt(1/(1- sum(D_mybest(newgam,:).^2)));
 D_mybest(:,k)=temp_F_k_k*(D(:,newgam)-D_mybest*D_mybest(newgam,:).');
   a_F(k)=temp_F_k_k*projections(newgam);
projections = projections -D_mybest(:,k)*a_F(k);
  normr2 = normr2 -a_F(k)^2;
 end
 tout(k) = toc(tstart);
 k = k + 1;
end
tstart = tic;
F(1:(k-1),1:(k-1))=linsolve(chol(D(gamma(1:(k-1)),gamma(1:(k-1)))),eye(k-1),solveropts);
time_for_F =  toc(tstart);
tout(k-1)=tout(k-1)+time_for_F;
%The time to compute F is added to the time for the (k-1)-th iteration, i.e., the last iteration.
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));

 


function [s,tout] = ForTimeProposedv1(x, dict, D, dictTx, natom, tolerance)
clear dict;
% Actually dict, i.e., the dictionary, is not required.
% Proposed Memory-Saving version 1. 
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize =  size(D,2); %size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
tout = zeros(natom,1);
solveropts.UT = true;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
k = 1;
solveropts.UT = true;
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 tstart = tic;
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   a_F(1)= projections(newgam);
  projections = projections -D(:,newgam)*a_F(1);
   normr2 = normr2 -a_F(1)^2;
 else
      s_k_minus_1 = D(gamma(1:k-1),newgam);
      c_k_minus_1 = F(1:k-1,1:k-1)'*s_k_minus_1;
temp_F_k_k =sqrt(1/(1- sum(c_k_minus_1.^2)));
   F(1:k-1,k)=-temp_F_k_k*(F(1:k-1,1:k-1)*c_k_minus_1);
      F(k,k)=temp_F_k_k;
   a_F(k)=temp_F_k_k*projections(newgam);
projections = projections - D(:,gamma(1:k))*(F(1:k,k)*a_F(k));
  normr2 = normr2 -a_F(k)^2;
 end
 tout(k) = toc(tstart);
 k = k + 1;
end
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));






function [s,tout] = ForTimeProposedv2(x, dict, D, dictTx, natom, tolerance)
clear D;
% Actually D=dict'*dict, i.e., the Gram matrix of the dictionary, is not required.
% Proposed Memory-Saving version 2. 
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
tout = zeros(natom,1);
solveropts.UT = true;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
k = 1;
solveropts.UT = true;
D_mybest = zeros(dictsize,natom);
temp_D_row = zeros(natom,1);
E = zeros(size(dict,1),natom);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 tstart = tic;
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   D_mybest(:,1) = dict.'*dict(:,newgam);
   E(:,1)=dict(:,newgam);
   a_F(1)= projections(newgam);
  projections = projections -D_mybest(:,1)*a_F(1);
   normr2 = normr2 -a_F(1)^2;
 else
temp_D_row = D_mybest(newgam,:).';
temp_F_k_k =sqrt(1/(1- sum(temp_D_row.^2)));
 E(:,k)=temp_F_k_k*(dict(:,newgam)-E*temp_D_row);
 D_mybest(:,k)=dict.'*E(:,k);
   a_F(k)=temp_F_k_k*projections(newgam);
projections = projections -D_mybest(:,k)*a_F(k);
  normr2 = normr2 -a_F(k)^2;
 end
 tout(k) = toc(tstart);
 k = k + 1;
end
tstart = tic;
F(1:(k-1),1:(k-1))=linsolve(chol(dict(:,gamma(1:(k-1))).'*dict(:,gamma(1:(k-1)))),eye(k-1),solveropts);
time_for_F =  toc(tstart);
tout(k-1)=tout(k-1)+time_for_F;
%The time to compute F is added to the time for the (k-1)-th iteration, i.e., the last iteration.
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));

 


function [s,tout] = ForTimeProposedv3(x, dict, D, dictTx, natom, tolerance)
clear D;
% Actually D=dict'*dict, i.e., the Gram matrix of the dictionary, is not required.
% Proposed Memory-Saving version 3
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
tout = zeros(natom,1);
solveropts.UT = true;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
k = 1;
solveropts.UT = true;
temp_D_row = zeros(natom,1);
E = zeros(size(dict,1),natom);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 tstart = tic;
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   E(:,1)=dict(:,newgam);
   a_F(1)= projections(newgam);
projections = projections -dict.'*(E(:,1)*a_F(1));
   normr2 = normr2 -a_F(1)^2;
 else
temp_D_row = (dict(:,newgam).'*E).';
temp_F_k_k =sqrt(1/(1- sum(temp_D_row.^2)));
 E(:,k)=temp_F_k_k*(dict(:,newgam)-E*temp_D_row);
   a_F(k)=temp_F_k_k*projections(newgam);
projections = projections -dict.'*(E(:,k)*a_F(k));
  normr2 = normr2 -a_F(k)^2;
 end
 tout(k) = toc(tstart);
 k = k + 1;
end
tstart = tic;
F(1:(k-1),1:(k-1))=linsolve(chol(dict(:,gamma(1:(k-1))).'*dict(:,gamma(1:(k-1)))),eye(k-1),solveropts);
time_for_F =  toc(tstart);
tout(k-1)=tout(k-1)+time_for_F;
%The time to compute F is added to the time for the (k-1)-th iteration, i.e., the last iteration.
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));


function [s,tout] = ForTimeProposedv4(x, dict, D, dictTx, natom, tolerance)
clear D;
% Actually D=dict'*dict, i.e., the Gram matrix of the dictionary, is not required.
% Proposed Memory-Saving version 4
% The code is obtained by modifying the relevant code in [1].
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% INPUTS
% x        - input signal
% dict     - dictionary
% D        - dictionary Gramian
% dictTx    - dict'*x
% natom    - stopping crit. 1: max number of iterations
% tolerance - stopping crit. 2: minimum norm residual
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
% OUTPUTS
% s  - solution to x = dict*s
%
%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%=%
x = x(:);
% initialization
residual = x;
normx2 = x'*x;
normtol2 = tolerance*normx2;
normr2 = normx2;
dictsize = size(dict,2);
L = zeros(natom,natom);
gamma = zeros(natom,1);
% find initial projections
projections = dictTx;
tout = zeros(natom,1);
solveropts.UT = true;
F=eye(natom);
a_F = zeros(natom,1);
temp_F_k_k = 0;
k = 1;
solveropts.UT = true;
temp_D_row = zeros(natom,1);
s = zeros(dictsize,1);
while normr2 > normtol2 && k <= natom
 tstart = tic;
 [alpha,maxindex] = max(projections.^2);
 newgam = maxindex(1);
 gamma(k) = newgam;
 if k==1
   a_F(1)= projections(newgam);
projections = projections -dict.'*(dict(:,gamma)*(F(1,1)*a_F(1)));
   normr2 = normr2 -a_F(1)^2;
 else
g_lamda_k = dict(:,newgam).'*dict(:,newgam);
s_k_minus_1 = dict(:,gamma(1:k-1)).'*dict(:,newgam);
c_k_minus_1 = F(1:k-1,1:k-1)'*s_k_minus_1;
temp_F_k_k =sqrt(1/(1- sum(c_k_minus_1.^2)));
F(:,k)=-temp_F_k_k*(F*c_k_minus_1);
F(k,k)=temp_F_k_k;
a_F(k)=temp_F_k_k*projections(newgam);
projections = projections -dict.'*(dict(:,gamma)*(F(1:k,k)*a_F(k)));
  normr2 = normr2 -a_F(k)^2; 
 end
 tout(k) = toc(tstart);
 k = k + 1;
end
tstart = tic;
%The time to compute F is added to the time for the (k-1)-th iteration, i.e., the last iteration.
s(gamma(1:(k-1)))=F(1:(k-1),1:(k-1))*a_F(1:(k-1));

 

 

 







 

 




 


