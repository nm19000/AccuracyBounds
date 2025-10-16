function u = img_get_PSF(rx,ry,...
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
                         f_cut,dom)

% get pupil function
u = img_get_PPL(rx,ry,f_cut,fx_mid,fy_mid,Kx,Ky,fx_min,fy_min,fx_max,fy_max,dx,dy);
% (fx_max-fx_min)/Kx*(fy_max-fy_min)/Ky * sum(abs(u).^2,"all")

if strcmp(dom,'f')
    return
end

if strcmp(dom,'F')
    u = [[u;zeros(Rx-Kx,Ky)],zeros(Rx,Ry-Ky)];
    return
end

% transform
u = ifft2( Rx/Kx*Ry/Ky*(fx_max-fx_min)*(fy_max-fy_min)*u , Rx,Ry );
% (gx_max-gx_min)/Qx*(gy_max-gy_min)/Qy * sum(abs(u).^2,"all")


if strcmp(dom,'G')
    u = abs(u).^2;
    return
end


% reduce
u = abs( u(1:Qx,1:Qy) ).^2;

if strcmp(dom,'g')
    % return with re-normalization so it compares with total-image-unit
    % u = (gx_max-gx_min)/Qx*(gy_max-gy_min)/Qy * Ax*Ay*u;
    return
end

% integrate
u = (gx_max-gx_min)/Qx*(gy_max-gy_min)/Qy * shiftdim(reshape(sum(reshape(shiftdim(reshape(sum(reshape(u,Qx/Px,Px*Qy),1),Px,Qy),1),Qy/Py,Py*Px),1),Py,Px),1);

if strcmp(dom,'r')
    return
end

error('Unknown domain')
