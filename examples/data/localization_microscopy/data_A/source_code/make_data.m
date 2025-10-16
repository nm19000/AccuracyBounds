clear
clc
    
[w_cnt,u_cnt,...
 wM,wV,wG,wF,t_exp,S_pxl,lambda,nNA,...
 units_length,units_time,units_image,...
 ground] = generate_synthetic_data();

opts.w_cnt = w_cnt;
opts.wM = wM;
opts.wV = wV;
opts.wG = wG;
opts.wF = wF;
opts.t_exp = t_exp;
opts.ds = S_pxl;
opts.lambda = lambda;
opts.nNA = nNA;
opts.units_length = units_length;
opts.units_time = units_time;
opts.units_image = units_image;...
opts.ground = ground;

chain = chainer_main(   [],   0, opts,false,false);
chain = chainer_main(chain,25, opts,false,false);
idx = find( chain.i > 0.4*chain.i(end) );

%% reduce data

v_bar = wV/wG^2;
f_bar = wF;

w = (w_cnt-wM)/wG;
u = u_cnt;
e = (w-u)./sqrt(v_bar+f_bar*u);

x_grnd = ground.x;
y_grnd = ground.y;
C_grnd = ground.C;
h_grnd = ground.h;

x_mcmc = double( chain.x(idx) );
y_mcmc = double( chain.y(idx) );
C_mcmc = double( chain.C(idx) );
h_mcmc = double( chain.h(idx) );

P_mcmc = double( chain.P(1,idx) );
L_mcmc = double( chain.P(2,idx) );

x_mcmc_mean = mean(x_mcmc);
y_mcmc_mean = mean(y_mcmc);
C_mcmc_mean = mean(C_mcmc);
h_mcmc_mean = mean(h_mcmc);

x_mcmc_median = median(x_mcmc);
y_mcmc_median = median(y_mcmc);
C_mcmc_median = median(C_mcmc);
h_mcmc_median = median(h_mcmc);

[~,j] = max(P_mcmc);
x_mcmc_map = x_mcmc(j);
y_mcmc_map = y_mcmc(j);
C_mcmc_map = C_mcmc(j);
h_mcmc_map = h_mcmc(j);

[~,j] = max(L_mcmc);
x_mcmc_ml = x_mcmc(j);
y_mcmc_ml = y_mcmc(j);
C_mcmc_ml = C_mcmc(j);
h_mcmc_ml = h_mcmc(j);




%%
function [w_cnt,u_cnt,...
          wM,wV,wG,wF,t_exp,S_pxl,lambda,nNA,...
          units_length,units_time,units_image,...
          ground] = generate_synthetic_data()

% setup
units_length = '\mum';
units_time   =  'sec';
units_image  =  'adu';

% data sizes: 1st dimension, 2nd dimension, and exposures
Px = 15;
Py = 16;
N  =  1;

% camera read-out: offset, variance, overall gain, excess noise factor
wM = 200;       % [image]
wV =  10;       % [image]^2
wG =  10;       % [image]/[photon]
wF =   2;       % [photon]

% exposure period
t_exp = 0.100; % [time]

% intensities: background flux and emitter brightness
C_bck = 1e4; % [photons]/[time]/[length]^2
h_brg = 1e5; % [photons]/[time]

% physical pixel width
s_pxl = 13.3; % [length]

% optics: numerical apperture, wavelength, and magnification
M = 100;
nNA = 1.45;
lambda = 0.550; % [length]



%% optics
S_pxl = s_pxl/M; % actual pixel size
f_cut = nNA/lambda;

%% generate position
x = -S_pxl/2 + S_pxl*rand; % [length]
y = -S_pxl/2 + S_pxl*rand; % [length]

%% image ranges
gx_min = -0.5*S_pxl*Px;
gx_max = +0.5*S_pxl*Px;

gy_min = -0.5*S_pxl*Py;
gy_max = +0.5*S_pxl*Py;

%% grids
[Qx,Qy,...
 Kx,Ky,...
 Rx,Ry,...
 dx,dy,...
 fx_min,fx_max,...
 fy_min,fy_max,...
 fx_mid,fy_mid,...
 ~,~,...
 ~,~,...
 ~,~,...
 ~,~,...
 ~,~,...
 ~,~] = img_get_grids(Px,Py,...
                      gx_min,gx_max,...
                      gy_min,gy_max,...
                      2*f_cut,...
                      false);

%% generate images
v_psf = img_get_PSF(x,y,...
                    dx,dy,...
                    Px,Py,...
                    Qx,Qy,...
                    Kx,Ky,...
                    Rx,Ry,...
                    fx_mid,fy_mid,...
                    gx_min,gy_min,...
                    gx_max,gy_max,...
                    fx_min,fy_min,...
                    fx_max,fy_max,...
                    f_cut,'r');

u_cnt = t_exp * ( C_bck * S_pxl^2 + h_brg * v_psf );

% get read-out
w_cnt = wM + wG*u_cnt + sqrt(wV+wF*wG^2*u_cnt).*randn(Px,Py,N);

%% save ground
ground.x = x;
ground.y = y;
ground.C = C_bck;
ground.h = h_brg;

end