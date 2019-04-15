%% Problem 1a - Adapted from HW2
% BME 593. Computational Methods For Imaging Science
%
% Michael Walker 4/14/2019

%% Setup
clear;
tic;
printcomment = @(varargin)fprintf('%-60s %5.1fs\n',sprintf(varargin{:}),toc);

%% Load data
printcomment('Loading data...');
load project_data
%%{
% Decimate 2x
sinogram(:,2:2:end) = [];
sinogram(2:2:end,:) = [];
imgref(:,2:2:end) = [];
imgref(2:2:end,:) = [];
%}

%%
N = size(imgref,1);          % Image is N-by-N pixels
theta = linspace(0,225,size(sinogram,2)+1); % projection angles
theta(end) = [];
p = size(sinogram,1);

% Assemble the X-ray tomography matrix, the true data, and true image
printcomment('Constructing forward operator...');
%[K, d, m_true] = paralleltomo(N, theta, p);
K = paralleltomo(N, theta, p,N*(p-1)/p);

%{
% Scale things. But the reference image is not scaled?
L = 0.06144;
pixel_size = L/N;
K = K*pixel_size;
imgref = imgref*pixel_size;
%}

%%
m_true = imgref(:);
d = sinogram(:);

printcomment('Plot true image and data...');
m_plt = m_true;
figure(1)
clf;
subplot(121);
imagesc(reshape(m_plt, N, N));
title('True image');
axis image
colorbar
subplot(122);
imagesc(reshape(d, p, length(theta)));
title('Data (sinograph)');
axis image
colorbar

printcomment('Purging rows');
% Remove possibly 0 rows from K and d
[K, d] = purge_rows(K, d);

%% Reconstruct m using SART. 
% Report convergence history and compare with ART

Kt = K.';
Ksqrnrminv = 1./sum(Kt.^2);

q = size(K,1);

Ktwts = Kt.*(1/q*Ksqrnrminv);

m_last = zeros(size(K,2),1);
m_last = K.'*d;
res = zeros(1e3,1);
err = res;
printcomment('SART iterations...');
for it = 1:1e3
    
    m_last = m_last + Ktwts*(d - K*m_last);
    res(it) = norm(m_last.'*Kt - d.');
    err(it) = norm(m_true - m_last);
    if mod(it,100)==0
        printcomment('  iteration %d',it);
        %%{
        m_plt = m_last;
        figure(2)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title(sprintf('Current Estimate, it %d',it));
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        snapnow;
        %}
    end
end
printcomment('  done.');

res_sart = res;
err_sart = err;
%%
figure(3)
clf
semilogy(res_sart);
hold on
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err_sart,'--');
xlabel('Iteration j');
ylabel('Norm');
legend('SART Residual','SART Error');

%% C) Reconstruct m using SIRT. 
% Report convergence history and compare with ART and SART
Ktwts = Kt.*(2/q*Ksqrnrminv);

m_last = zeros(size(K,2),1);
m_last = K.'*d;
res = zeros(1e3,1);
err = res;
printcomment('SIRT iterations...');
for it = 1:1e3
    
    m_last = m_last + Ktwts*(d - K*m_last);
    res(it) = norm(m_last.'*Kt - d.');
    err(it) = norm(m_true - m_last);
    if mod(it,100)==0
        printcomment('  iteration %d',it);
        %%{
        m_plt = m_last;
        figure(8)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title(sprintf('Current Estimate, it %d',it));
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        snapnow;
        %}
    end
    
    
end
printcomment('  done.');

res_sirt = res;
err_sirt = err;

figure(3)
clf
semilogy(res_sart);
hold on
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err_sart,'--');
semilogy(res_sirt);
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err_sirt,'--');
xlabel('Iteration j');
ylabel('Norm');
legend('SART Residual','SART Error','SIRT Residual','SIRT Error');

