function P = get_log_probs(x,y,h,C,params)
% P = log( priors )

if nargin<1
    P = 2;
    return
end

if x<params.x_prior_min || x>params.x_prior_max ...
|| y<params.y_prior_min || y>params.y_prior_max ...
|| h<0 || C<0
    error('Prior bounds are violated')
end


P = zeros(get_log_probs,1);

%% Priors

% h
P(1) = (params.h_prior_phi-1)*log(h/params.h_prior_ref) - params.h_prior_phi*h/params.h_prior_ref;

% C
P(2) = (params.C_prior_phi-1)*log(C/params.C_prior_ref) - params.C_prior_phi*C/params.C_prior_ref;

