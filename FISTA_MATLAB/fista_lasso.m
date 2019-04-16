function [X data_list] = fista_lasso(Y, D, Xinit, opts,m_true)

    if ~isfield(opts, 'backtracking')
        opts.backtracking = false;
    end 

    opts = initOpts(opts);
    lambda = opts.lambda;

    if numel(lambda) > 1 && size(lambda, 2)  == 1
        lambda = repmat(opts.lambda, 1, size(Y, 2));
    end
    if numel(Xinit) == 0
        Xinit = zeros(size(D,2), size(Y,2));
    end
    %% cost f
    function cost = calc_f(X)
        cost = 1/2 *normF2(Y - D*X);
    end 
    %% cost function 
    function cost = calc_F(X)
        if numel(lambda) == 1 % scalar 
            cost = calc_f(X) + lambda*norm1(X);
        elseif numel(lambda) == numel(X)
            cost = calc_f(X) + norm1(lambda.*X);
        end
    end 
    %% gradient
  
    DtY = D'*Y;
    function res = grad(X) 
        DxX =D*X;
        res = D' * DxX - DtY;
        disp('gradient done');
    end 
    %% Checking gradient 
    if opts.check_grad
        check_grad(@calc_f, @grad, Xinit);
    end 

    opts.max_iter = 50;
    %% Lipschitz constant 
    % L = max(eig(DtD));
    % a more effcient way to get eigen value
    
%     s = max(svds(D));
%     L = s;
%     disp('Lipschitz done');
    %% Use fista 
   % [X, ~, ~] = fista_general(@grad, @proj_l1, Xinit, 0.0153, opts, @calc_F);
  [X, iter, data_list] = my_fista(@grad, @proj_l1, Xinit, 0.0153, opts, @calc_F, D , Y,m_true);
end 