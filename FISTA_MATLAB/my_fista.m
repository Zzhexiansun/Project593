function [X, iter, data_list] = my_fista(grad, proj, Xinit, L, opts, calc_F, K , d,m_true)
% new items
%  K : forward model matrix, to compute the residual
%  d : the data
%  m_true
% data_list.residual 
% data_list.error 
% data_list.penalty 

    if ~isfield(opts, 'max_iter')
        opts.max_iter = 500;
    end
    if ~isfield(opts, 'regul')
        opts.regul = 'l1';
    end     
    if ~isfield(opts, 'pos')
        opts.pos = false;
    end
    
    if ~isfield(opts, 'tol')
        opts.tol = 1e-8;
    end
    
    if ~isfield(opts, 'verbose')
        opts.verbose = false;
    end
    Linv = 1/L;    
    lambdaLiv = opts.lambda*Linv;
    % opts_shrinkage = opts;
    % opts_shrinkage.lambda = lambdaLiv;
    x_old = Xinit;
    y_old = Xinit;
    t_old = 1;
    iter = 0;
    cost_old = 1e10;
    %% MAIN LOOP
    
    opts_proj = opts;
    opts_proj.lambda = lambdaLiv;
    while  iter < opts.max_iter
        iter = iter + 1;
        x_new = feval(proj, y_old - Linv*feval(grad, y_old), opts_proj);
        t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
        %% check stop criteria
        e = norm1(x_new - x_old)/numel(x_new);
        if e < opts.tol
            break;
        end
        %% update
        x_old = x_new;
        t_old = t_new;
        y_old = y_new;
        
        %% generate cost history
        % residual ||km - d ||
        data_list.residual(iter)  = norm1(K*x_new - d);
        % error || m - m_true||
        data_list.error(iter) = norm1(x_new -m_true);
        % TV penalty term TV(m)
        data_list.penalty(iter) = norm1(x_new);
        %% show progress
        if opts.verbose
            if nargin ~= 0
                cost_new = feval(calc_F, x_new);
                if cost_new <= cost_old 
                    stt = 'YES.';
                else 
                    stt = 'NO, check your code.';
                end
                fprintf('iter = %3d, cost = %f, cost decreases? %s\n', ...
                    iter, cost_new, stt);
                cost_old = cost_new;
            else 
                if mod(iter, 5) == 0
                    fprintf('.');
                end
                if mod(iter, 10) == 0 
                   fprintf('%d', iter);
                end     
            end        
        end 
    end
    X = x_new;
    if nargout == 3 
        min_cost = feval(calc_F, X);
    end 
end 