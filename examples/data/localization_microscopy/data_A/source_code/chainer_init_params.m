function params = chainer_init_params(opts)

%% Set-up

% units
params.units_length = opts.units_length;
params.units_time   = opts.units_time;
params.units_image  = opts.units_image;

% readout
params.wM = opts.wM; % [image]          read-out offset
params.wV = opts.wV; % [image]^2        read-out variance
params.wG = opts.wG; % [image]/[photon] overall gain
params.wF = opts.wF; % [photon]         excess noise factor

% sizes
params.Px = size(opts.w_cnt,1);
params.Py = size(opts.w_cnt,2);
params.N  = size(opts.w_cnt,3);

%% nominal optics
params.f_nom = opts.nNA/opts.lambda; % 1/[length] radius

%% other aquisition parameters
params.t_exp = opts.t_exp;
params.ds    = opts.ds;

%% edges
params.gx_min = -0.5*params.ds*params.Px;
params.gx_max = +0.5*params.ds*params.Px;

params.gy_min = -0.5*params.ds*params.Py;
params.gy_max = +0.5*params.ds*params.Py;

%% grids
[params.Qx,params.Qy,...
 params.Kx,params.Ky,...
 params.Rx,params.Ry,...
 params.dx,params.dy,...
 params.fx_min,params.fx_max,...
 params.fy_min,params.fy_max,...
 params.fx_mid,params.fy_mid,...
 params.fx_bnd,params.fy_bnd,...
 params.gx_mid,params.gy_mid,...
 params.gx_bnd,params.gy_bnd,...
 params.Fx_bnd,params.Fy_bnd,...
 params.Gx_bnd,params.Gy_bnd,...
 params.rx_bnd,params.ry_bnd] = img_get_grids(params.Px,params.Py,...
                                              params.gx_min,params.gx_max,...
                                              params.gy_min,params.gy_max,...
                                              2*params.f_nom,...
                                              false);

%% pre-processed measurments
params.dW_cnt = (opts.w_cnt-params.wM)/params.wG; % [photons]

%% positions
params.x_prior_min = params.gx_min; 
params.x_prior_max = params.gx_max;

params.y_prior_min = params.gy_min; 
params.y_prior_max = params.gy_max;


%% photon emission rates
params.h_prior_phi = 1;
params.h_prior_ref = abs( max(params.dW_cnt,[],[1,2,3])/params.t_exp * 3 ); % [photons]/[time]

params.C_prior_phi = 1;
params.C_prior_ref = abs( sum(params.dW_cnt,'all')/(params.N*params.t_exp)/(params.ds^2*params.Px*params.Py) ); % [photons]/[area]/[time]

%% ground
if isfield(opts,'ground')
    params.ground = opts.ground;
end

%% MCMC Samplers

params.rep_xy = 3;
params.rep_hC = 3;

r_sig = 0.5/params.f_nom; % Abbe diff limit of nominal optics
params.sig_X = 0.5*r_sig;
params.sig_Y = 0.5*r_sig;

params.sig_H = 0.25;
params.sig_c = 0.25;

