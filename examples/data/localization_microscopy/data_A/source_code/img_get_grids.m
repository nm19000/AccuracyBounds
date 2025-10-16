function [Qx,Qy,...
          Kx,Ky,...
          Rx,Ry,...
          dx,dy,...
          fx_min,fx_max,...
          fy_min,fy_max,...
          fx_mid,fy_mid,...
          fx_bnd,fy_bnd,...
          gx_mid,gy_mid,...
          gx_bnd,gy_bnd,...
          Fx_bnd,Fy_bnd,...
          Gx_bnd,Gy_bnd,...
          rx_bnd,ry_bnd] = img_get_grids(Px,Py,...
                                         gx_min,gx_max,...
                                         gy_min,gy_max,...
                                         bar_f_low,...
                                         fl)

bar_gx = gx_max - gx_min;
bar_gy = gy_max - gy_min;

Lx = ceil( bar_f_low*bar_gx );
Ly = ceil( bar_f_low*bar_gy );
% Lx = 2^nextpow2( Lx ); % ???
% Ly = 2^nextpow2( Ly ); % ???

bar_fx = Lx/bar_gx;
bar_fy = Ly/bar_gy;

Ax = 8; % adjust for up-sampling here
Ay = 8; % adjust for up-sampling here

Bx = 8; % adjust for up-sampling here
By = 8; % adjust for up-sampling here

Qx = Ax*Px;
Qy = Ay*Py;

Kx = Bx*Lx;
Ky = By*Ly;

Rx = Ax*Bx*Px; %  = Kx_prime = Qx_prime
Ry = Ay*By*Py; %  = Ky_prime = Qy_prime

fx_min = -0.5*bar_fx;
fy_min = -0.5*bar_fy;

fx_max = +0.5*bar_fx;
fy_max = +0.5*bar_fy;

dx = (fx_max-fx_min)/Kx*(gx_min+0.5*(gx_max-gx_min)/Qx);
dy = (fy_max-fy_min)/Ky*(gy_min+0.5*(gy_max-gy_min)/Qy);

gx_bnd = reshape( linspace(gx_min,gx_max,Qx+1), Qx+1, 1);
gy_bnd = reshape( linspace(gy_min,gy_max,Qy+1), 1, Qy+1);

fx_bnd = reshape( linspace(fx_min,fx_max,Kx+1), Kx+1, 1);
fy_bnd = reshape( linspace(fy_min,fy_max,Ky+1), 1, Ky+1);

rx_bnd = reshape( linspace(gx_min,gx_max,Px+1), Px+1, 1);
ry_bnd = reshape( linspace(gy_min,gy_max,Py+1), 1, Py+1);

gx_mid = 0.5*(gx_bnd(1:end-1)+gx_bnd(2:end));
gy_mid = 0.5*(gy_bnd(1:end-1)+gy_bnd(2:end));

fx_mid = 0.5*(fx_bnd(1:end-1)+fx_bnd(2:end));
fy_mid = 0.5*(fy_bnd(1:end-1)+fy_bnd(2:end));


Fx_bnd = reshape( linspace(fx_min,fx_min+Rx/Kx*bar_fx,Rx+1), Rx+1, 1);
Fy_bnd = reshape( linspace(fy_min,fy_min+Ry/Ky*bar_fy,Ry+1), 1, Ry+1);

Gx_bnd = reshape( linspace(gx_min,gx_min+Rx/Qx*bar_gx,Rx+1), Rx+1, 1);
Gy_bnd = reshape( linspace(gy_min,gy_min+Ry/Qy*bar_gy,Ry+1), 1, Ry+1);


%%
if fl
    disp('--- Grid ')
    disp(['Lx = ',num2str(Lx,'%5d'), ' - Rx = ',num2str(Rx,'%5d') ])
    disp(['Ly = ',num2str(Ly,'%5d'), ' - Ry = ',num2str(Ry,'%5d') ])
end

