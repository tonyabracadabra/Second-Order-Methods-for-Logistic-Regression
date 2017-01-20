function [f_diff,w] = IRLS_barrier(X,y,alpha,beta)
    mumu = 20;
    [n m] = size(X);
    X = [ones(n,1), X];
    
    % Find start feasible point
    f = [zeros(1,2*m+1),ones(1,m+1)];
    AIneq = [ones(1,m+1),-ones(1,m),-ones(1,m+1); ...
            -ones(1,m+1),-ones(1,m),-ones(1,m+1)];
    bIneq = [0;0];
    x = linprog(f,AIneq,bIneq);
%     w = x(1:m+1);
    w = zeros(m+1,1);
%     u = x(m+2:2*m+1);
    u = 0.01*ones(m,1);
    
    lambda = 15;

    mu = @(w) 1./(1+exp(-X*w));
    f = @(w,u) -y'*X*w+sum(log(exp(X*w)+1))+lambda*sum(u);
    phi = @(w,u) -sum(log(u.^2-w(2:end).^2));
    g = @(t,w,u) t*f(w,u) + phi(w,u);
    gd_w = @(t,w,u) t*X'*(mu(w)-y)+[0;2*w(2:end)./(u.^2-w(2:end).^2)];
    gd_u = @(t,w,u) t*lambda-2*(u./(u.^2-w(2:end).^2));
    
    % Hessian for function f
    H_f_w = @(w,u) X'*diag(mu(w).*(1-mu(w)))*X;
    H_f = @(w,u) [H_f_w(w,u), zeros(m+1,m);zeros(m,2*m+1)];
    
    % ((u-w(2:end)).^2).*((u+w(2:end)).^2))
    
    % Hessian for function phi into four blocks
    % Hessian beta(1:p),beta(1:p), phi
    B1 = @(w,u) diag(2*(u.^2+w(2:end).^2)./(u.^2-w(2:end).^2).^2);
    % Hessian beta(1:p),u, phi
    B2 = @(w,u) diag(-4*(w(2:end).*u./(u.^2-w(2:end).^2).^2));
    % Hessian u,beta(1:p) phi
    B3 = @(w,u) diag(-4*(w(2:end).*u./(u.^2-w(2:end).^2).^2));
    % Hessian u,u, phi
    B4 = @(w,u) diag(2*(u.^2+w(2:end).^2)./(u.^2-w(2:end).^2).^2);
    
    H_phi = @(w,u) [zeros(1,2*m+1);zeros(m,1),B1(w,u),B2(w,u);zeros(m,1),B3(w,u),B4(w,u)];

    A = @(t,w,u) t*H_f(w,u) + H_phi(w,u);
    B = @(t,w,u) [gd_w(t,w,u);gd_u(t,w,u)];
   
    f_optimal = 306.476;

    k = 0;
    t_outer = 5;
%     v = -A(t_outer,w,u)\B(t_outer,w,u);
    fs = [f(w,u)];
    
    while 2*m/t_outer > 1e-9
        disp(t_outer);
        gs = [g(t_outer,w,u)];
        while true
            t_inner = 1;

            v = -A(t_outer,w,u)\B(t_outer,w,u);
            tmp = alpha*B(t_outer,w,u)'*v;

            while g(t_outer,w+t_inner*v(1:m+1),u+t_inner*v(m+2:end))>g(t_outer,w,u)+ t_inner*tmp ...
                   || sum((u+t_inner*v(m+2:end)) < abs(w(2:end)+t_inner*v(2:m+1))) ~= 0
                t_inner = beta*t_inner;
            end
                
            w = w + t_inner*v(1:m+1);
            u = u + t_inner*v(m+2:end);

            fs = [fs f(w,u)];
            gs = [gs g(t_outer,w,u)];

            diff = gs(end-1) - gs(end);
            if diff < 1e-9
                break
            end
            k = k + 1;
        end
        t_outer = mumu*t_outer;
    end
    f_diff = fs - f_optimal;
end
