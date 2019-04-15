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
clim = [0,max(abs(imgref(:)))];
%%
N = size(imgref,1);          % Image is N-by-N pixels
theta = linspace(0,225,size(sinogram,2)+1); % projection angles
theta(end) = [];
p = size(sinogram,1);

% Assemble the X-ray tomography matrix, the true data, and true image
printcomment('Constructing forward operator...');
%[K, d, m_true] = paralleltomo(N, theta, p);
K = paralleltomo(N, theta, p,N*(p-1)/p);

%%{
% Scale things. But the reference image is not scaled?
L = 0.06144;
pixel_size = L/N;
K = K*pixel_size;
%imgref = imgref*pixel_size;
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


%% A) Reconstruct m using ART.
% Report convergence history (i.e. residual and error norms at each iteration)

Kt = K.';
Ksqrnrminv = 1./sum(Kt.^2);

q = size(K,1);

m_last = zeros(size(K,2),1);
res = zeros(1e3,1);
err = res;
res0 = res;
err0 = res;
%%{
printcomment('ART iterations...');
for itt = 1:q*1e3
    i = mod(itt-1,size(K,1))+1;
    m_last = m_last + Ksqrnrminv(i)*(d(i) - m_last'*Kt(:,i))*Kt(:,i);
    
    % Computing metrics at each iteration is expensive. Compute for first 1000 iterations, then just the last iteration
    % in each swipe.
    if itt <= 1e3
        % First 1000 iterations (1st swipe)
        res0(itt) = norm(m_last.'*Kt - d.');
        err0(itt) = norm(m_true - m_last);
        if itt == 1e3
            printcomment('  iteration %d',itt);
        end
    end
    if mod(itt,q)==0
        % Last iteration of each swipe
        res(itt/q) = norm(m_last.'*Kt - d.');
        err(itt/q) = norm(m_true - m_last);
        if mod(itt,q*100)==0
            printcomment('  iteration %d',itt);
        end
        %%{
        m_plt = m_last;
        figure(1)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title('Current Estimate');
        axis image
        colorbar
        set(gca,'clim',clim);
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        set(gca,'clim',clim);
        %}
    end
end
printcomment('  done.');
res0_art = res0;
err0_art = err0;
res_art = res;
err_art = err;

figure(2)
clf
semilogy([1:1e3,(1:numel(res))*q],[res0;res]);
hold on
semilogy([1:1e3,(1:numel(res))*q],[err0;err]);
xlabel('Iteration j');
ylabel('Norm');
legend('ART Residual','ART Error');

%% Reconstruct m using SART. 
% Report convergence history and compare with ART

Kt = K.';
Ksqrnrminv = 1./sum(Kt.^2);

q = size(K,1);
Ktwts = Kt.*(1/q*Ksqrnrminv);
%Ktwts = Kt;

m_last = zeros(size(K,2),1);
m_last = K.'*d;
res = zeros(1e3,1);
err = res;
printcomment('SART iterations...');
for it = 1:5e3
    
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
        set(gca,'clim',clim);
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        set(gca,'clim',clim);
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
        set(gca,'clim',clim);
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        set(gca,'clim',clim);
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

