function [f_diff] = IRLS(X,y,alpha,beta)

[n m] = size(X);
X = [ones(n,1), X];
w = zeros(m+1, 1);

mu = @(w) 1./(1+exp(-X*w));
f = @(w) -y'*X*w+sum(log(exp(X*w)+1));
gd = @(w) X'*(mu(w)-y);
H = @(w) X'*diag(mu(w).*(1-mu(w)))*X;
v = @(w) -H(w)\gd(w);
f_optimal = 186.637;

k = 0;
fs = [f(w)];
while true
    t = 1;
    update = v(w);
    tmp = alpha*gd(w)'*v(w);
    while f(w+t*update)>f(w)+t*tmp
        t = beta*t;
    end
    w_new = w + t*update;
    fs = [fs f(w_new)];
    diff = fs(end) - fs(end-1);
    if abs(diff) < 1e-6
        break
    end
    k = k + 1;
    w = w_new;
end
f_diff = fs - f_optimal;

end
