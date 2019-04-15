clear; clc; close all;

% import data.
% variable name: sinogram is sinogram
% variable name: imgref is the reference image.
load project_data.mat ;
% downsample rate
% only downsample the angle.
% dsratio downsamples the views of sinogram
% dsratio1 downsamples the pixel resolution in the image.
dsratio =2 ;
dsratio1 = 2;
% create fwd operator matrix
L = 0.06144;
npixels = 256/dsratio1;
pixel_size = L/npixels;
nviews = 540/dsratio;
dtheta = 5/12*dsratio;
views = [0:nviews-1]*dtheta;
nrays = 512/dsratio1;
d = npixels*(nrays-1)/nrays;
A = paralleltomo(npixels,views,nrays,d);
K = A * pixel_size;
clear A;
% K is the fwd operator.

d = sinogram ; 
m_true = imgref;

% down sample sinogram d 
if (dsratio>1)
    d = downsample(d,dsratio1);
    d = downsample(d',dsratio)';
end

% down sample the image m_ture;

if (dsratio1 > 1)
    m_true = downsample(m_true,dsratio1);
    m_true = downsample(m_true',dsratio1)';
end

% initial guess m0 = 0;

m0 = zeros(npixels^2,1);
% compute D
[q, sizeM] = size(K);

d = reshape(d,[q,1]);
m_true = reshape(m_true,[sizeM,1]);
%% start ART
for iter = 1:20
   for j = 1:q
       mj = m0 + ((d(j)-K(j,:)*m0)*K(j,:) / sum(K(j,:).^2))';
       m0 = mj;
   end
   ResidualART(iter) = sqrt(sum(sum((d - K*mj).^2)));
   ErrorART(iter) = sqrt(sum(sum((m_true - mj).^2)));   
   iter
end




%% SIRT 
parpool(2)
D = zeros(q,1);
parfor i = 1:q
    D(i) = 1./norm(K(i,:));
    if(rem(i,100)==0)
        i
    end
end
