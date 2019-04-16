clear; clc; close all;

% import data.
% variable name: sinogram is sinogram
% variable name: imgref is the reference image.
load project_data.mat ;
% downsample rate
% only downsample the angle.
% dsratio downsamples the views of sinogram
% dsratio1 downsamples the pixel resolution in the image.
dsratio =1 ;
dsratio1 = 1;
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

% set up intial guess m0 = 0;
m0 = zeros(npixels^2,1);

opts.lambda =0.1;
opts.max_iter = 500;
opts.tol = 1e-3;
opts.verbose = true;

d= reshape(d,[540*512/dsratio/dsratio1,1]);
m_true = reshape(m_true,[256*256,1]);

% start FISTA
[X, data_list] = fista_lasso(d, K, m0, opts,m_true);

%% compare the reconstruction and true image
figure(1)
subplot(121)
imshow(m_true,[])
title('true image')
subplot(122)
imshow(reshape(X,[256,256]),[]);
title('reconstruction')

%% L - curve 
figure(2)
loglog(data_list.residual,data_list.penalty);
title('L-curve')

%% converge history
figure(3)
subplot(131)
plot(data_list.residual)
title('||Km - d ||')
subplot(132)
plot(data_list.error)
title('||m - m-true||')
subplot(133)
plot(data_list.penalty)
title('||m||')