%% Assignment 2, Problem 2
% BME 593. Computational Methods For Imaging Science
%
% Michael Walker 3/2/19

%% Setup
clear;
printcomment = @(varargin)fprintf('%-60s %5.1fs\n',sprintf(varargin{:}),toc);
tic;

rng('default');

N = 64;          % Image is N-by-N pixels
theta = 0:2:178; % projection angles
p = 90;          % Number of rays for each angle

% Assemble the X-ray tomography matrix, the true data, and true image
[K, d, m_true] = paralleltomo(N, theta, p);

m_plt = m_true;
figure(1)
clf;
subplot(121);
imagesc(reshape(m_plt, N, N));
title('True image');
axis image
subplot(122);
imagesc(reshape(d, p, length(theta)));
title('Data (sinograph)');
axis image
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
        %{
        m_plt = m_last;
        figure(1)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title('Current Estimate');
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
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

%% B) Reconstruct m using SART. 
% Report convergence history and compare with ART
Ktwts = Kt.*(1/q*Ksqrnrminv);

m_last = zeros(size(K,2),1);
res = zeros(1e3,1);
err = res;
printcomment('SART iterations...');
for itt = 1:1e3
    
    m_last = m_last + Ktwts*(d - K*m_last);
    res(itt) = norm(m_last.'*Kt - d.');
    err(itt) = norm(m_true - m_last);
    if mod(itt,100)==0
        printcomment('  iteration %d',itt);
        %{
        m_plt = m_last;
        figure(1)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title('Current Estimate');
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        %}
    end
end
printcomment('  done.');

res_sart = res;
err_sart = err;

figure(3)
clf
semilogy(res0_art);
hold on
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err0_art,'--');
semilogy(res_sart);
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err_sart,'--');
xlabel('Iteration j');
ylabel('Norm');
legend('ART Residual','ART Error','SART Residual','SART Error');

%% C) Reconstruct m using SIRT. 
% Report convergence history and compare with ART and SART
Ktwts = Kt.*(2/q*Ksqrnrminv);

m_last = zeros(size(K,2),1);
res = zeros(1e3,1);
err = res;
printcomment('SIRT iterations...');
for itt = 1:1e3
    
    m_last = m_last + Ktwts*(d - K*m_last);
    res(itt) = norm(m_last.'*Kt - d.');
    err(itt) = norm(m_true - m_last);
    if mod(itt,100)==0
        printcomment('  iteration %d',itt);
        %{
        m_plt = m_last;
        figure(1)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title('Current Estimate');
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        %}
    end
    
    
end
printcomment('  done.');

res_sirt = res;
err_sirt = err;

figure(3)
clf
semilogy(res0_art);
hold on
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err0_art,'--');
semilogy(res_sart);
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err_sart,'--');
semilogy(res_sirt);
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1);
semilogy(err_sirt,'--');
xlabel('Iteration j');
ylabel('Norm');
legend('ART Residual','ART Error','SART Residual','SART Error','SIRT Residual','SIRT Error');

%%% Noise-Free Analysis
% Contrasting the methods we find ART reduces the residual (misfit) quickly at first, but does not significantly reduce
% the error over the first 1000 iterations. Over the first 1000 iterations SART and SIRT perform better than ART (SIRT
% performs better than SART) at about 2.5x the processing time of ART if we includ time required to compute the
% residual. With many many itartions, ART eventually performs well.

%% D) Consider the case with noisy data. 
% Reconstruct m using ART, SART, SIRT. Report convergence history and discuss what you observed.
printcomment('Adding noise.');
d = K*m_true;
figure(2)
set(gca,'ColorOrderIndex',1);
noise_level = 0.01; % noise level.
noise_std = noise_level*norm(d,'inf');
d = d + noise_std*randn(size(d));


%%% Art reconstruction on noisey data

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
    end
    if mod(itt,q)==0
        % Last iteration of each swipe
        res(itt/q) = norm(m_last.'*Kt - d.');
        err(itt/q) = norm(m_true - m_last);
        if mod(itt,q*100)==0
            printcomment('  iteration %d',itt);
        end
        %{
        m_plt = m_last;
        figure(1)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title('Current Estimate');
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        %}
    end
end
printcomment('  done.');
res0_art_n = res0;
err0_art_n = err0;
res_art_n = res;
err_art_n = err;

figure(4)
clf
semilogy([1:1e3,(1:numel(res))*q],[res0_art;res_art],'--');
hold on
semilogy([1:1e3,(1:numel(res))*q],[err0_art;err_art],'--');
set(gca,'ColorOrderIndex',1);
semilogy([1:1e3,(1:numel(res))*q],[res0;res]);
hold on
semilogy([1:1e3,(1:numel(res))*q],[err0;err]);
semilogy([1,numel(res)*q],noise_std*sqrt(q)*ones(1,2),'k');
xlabel('Iteration j');
ylabel('Norm');
legend({'ART Residual (Noise-free)','ART Error (Noise-free)','ART Residual','ART Error','$\sigma\sqrt{q}$'},...
    'interpreter','latex');

%% SART reconstruction on noisy data
Ktwts = Kt.*(1/q*Ksqrnrminv);

m_last = zeros(size(K,2),1);
res = zeros(1e3,1);
err = res;
printcomment('SART iterations...');
for itt = 1:1e3
    
    m_last = m_last + Ktwts*(d - K*m_last);
    res(itt) = norm(m_last.'*Kt - d.');
    err(itt) = norm(m_true - m_last);
    if mod(itt,1e2)==0
        printcomment('  iteration %d',itt);
        %{
        m_plt = m_last;
        figure(1)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title('Current Estimate');
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        %}
    end
end
printcomment('  done.');

res_sart_n = res;
err_sart_n = err;

%%% SIRT Reconstruction on noisy data

Ktwts = Kt.*(2/q*Ksqrnrminv);

m_last = zeros(size(K,2),1);
res = zeros(1e3,1);
err = res;
printcomment('SIRT iterations...');
for itt = 1:1e3
    
    m_last = m_last + Ktwts*(d - K*m_last);
    res(itt) = norm(m_last.'*Kt - d.');
    err(itt) = norm(m_true - m_last);
    if mod(itt,1e2)==0
        printcomment('  iteration %d',itt);
        %{
        m_plt = m_last;
        figure(1)
        clf;
        subplot(121);
        imagesc(reshape(m_plt, N, N));
        title('Current Estimate');
        axis image
        colorbar
        subplot(122);
        imagesc(reshape(abs(m_plt-m_true), N, N));
        title('Error');
        axis image
        colorbar
        %}
    end
end
printcomment('  done.');

res_sirt_n = res;
err_sirt_n = err;
%%
figure(5)
clf
semilogy([1,numel(res)],noise_std*sqrt(q)*ones(1,2),'k');
hold on

semilogy(res0_art,'--');
semilogy(err0_art,'--');
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-2);
semilogy(res0_art_n);
semilogy(err0_art_n);

semilogy(res_sart,'--');
semilogy(err_sart,'--');
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-2);
semilogy(res_sart_n);
semilogy(err_sart_n);

semilogy(res_sirt,'--');
semilogy(err_sirt,'--');
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-2);
semilogy(res_sirt_n);
semilogy(err_sirt_n);

xlabel('Iteration j');
ylabel('Norm');

%{
set(gca,'ColorOrderIndex',1);
semilogy((1:numel(res_art))*q,res_art,'--');
semilogy((1:numel(res_art))*q,err_art,'--');
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-2);
semilogy((1:numel(res_art))*q,res_art_n);
semilogy((1:numel(res_art))*q,err_art_n);
%}

legend({'$\sigma\sqrt{q}$','ART Residual (Noise-free)','ART Error (Noise-free)','ART Residual','ART Error',...
    'SART Residual (Noise-free)','SART Error (Noise-free)','SART Residual','SART Error',...
    'SIRT Residual (Noise-free)','SIRT Error (Noise-free)','SIRT Residual','SIRT Error'},...
    'interpreter','latex');



%%% Analysis
% For 1000 iterations, all methods seem to have strong noise immunity. ART seems to start diverging from the noise-free
% results by 30,000 iterations which is about 4 swipes.

%% E) Morozov discrepancy principle 
% Implement Morozov discrepancy principle as stopping criterion for ART
res = [res0_art_n;res_art_n];

idx = find(res_art_n > noise_std*sqrt(q),1,'last');

figure(6)
clf
loglog([1,numel(res_art)*q],noise_std*sqrt(q)*ones(1,2),'k');
hold on
loglog([1:1e3,(1:numel(res_art))*q],res)
loglog(idx*q,res_art_n(idx),'ro');
xlabel('iteration, j');
ylabel('l2 norm');
legend({'$\sigma\sqrt{q}$','ART Residual','Morozov'},'interpreter','latex');
title('Stopping Criteria');

figure(4)
clf
semilogy([1,numel(res_art)*q],noise_std*sqrt(q)*ones(1,2),'k');
hold on
semilogy([1:1e3,(1:numel(res_art))*q],[res0_art;res_art],'--');
semilogy([1:1e3,(1:numel(res_art))*q],[err0_art;err_art],'--');
set(gca,'ColorOrderIndex',1);
semilogy([1:1e3,(1:numel(res_art))*q],[res0_art_n;res_art_n]);
semilogy([1:1e3,(1:numel(res_art))*q],[err0_art_n;err_art_n]);
semilogy(idx*q*ones(1,2),[res_art_n(idx),err_art_n(idx)],'ro');
xlabel('Iteration j');
ylabel('Norm');
legend({'$\sigma\sqrt{q}$','ART Residual (Noise-free)','ART Error (Noise-free)','ART Residual','ART Error','Morozov'},...
    'interpreter','latex');
xlim([0,5e5])
grid on

%%% Analysis
% The Morozov discrepancy principle worked quite well in this case. The true error increased only slightly from the
% minimum (noisy reconstruction error) due to overfitting.
