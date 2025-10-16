function H = img_get_PPL(rx,ry,f_cut,fx_mid,fy_mid,Kx,Ky,fx_min,fy_min,fx_max,fy_max,dx,dy)

H = ( fx_mid.^2 ...
    + fy_mid.^2 <= f_cut^2 ) / ( sqrt(pi)*f_cut );

% apply translations
H = H .* exp( 2j*pi*( ( dx*(0:Kx-1)'-rx*fx_mid ) ...
                    + ( dy*(0:Ky-1) -ry*fy_mid ) ) );
