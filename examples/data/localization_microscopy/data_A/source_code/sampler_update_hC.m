function [sample_h,sample_C,sample_L,rec] = sampler_update_hC( ...
          sample_h,sample_C,sample_L,rec, x,y,params,beta)


V_cnt = params.t_exp*(params.gx_max-params.gx_min)/params.Px ...
                    *(params.gy_max-params.gy_min)/params.Py ;

v_cnt = params.t_exp*img_get_PSF(x,y,...
                                 params.dx,params.dy,...
                                 params.Px,params.Py,...
                                 params.Qx,params.Qy,...
                                 params.Kx,params.Ky,...
                                 params.Rx,params.Ry,...
                                 params.fx_mid,params.fy_mid,...
                                 params.gx_min,params.gy_min,...
                                 params.gx_max,params.gy_max,...
                                 params.fx_min,params.fy_min,...
                                 params.fx_max,params.fy_max,...
                                 params.f_nom,'r');


%% update
for rep = 1:sum(rand(3*params.rep_hC*2,1)<0.5)

    % prepare sampler
    sample_H = params.sig_H*randn;
    tempor_H = params.sig_H*randn;
    sample_c = params.sig_c*randn;
    tempor_c = params.sig_c*randn;

    % pick slice
    U_prop = log(rand);
    
    % pick interval
    T = 2*pi*rand;
    T_min = T - 2*pi;
    T_max = T;
    
    while true 

        rec(2) = rec(2) + 1;

        propos_H = cos(T)*sample_H + sin(T)*tempor_H;
        propos_c = cos(T)*sample_c + sin(T)*tempor_c;

        propos_h = sample_h*exp(sample_H-propos_H);
        propos_C = sample_C*exp(sample_c-propos_c);
    
        propos_L = get_log_like(propos_C*V_cnt...
                               +propos_h*v_cnt,params.dW_cnt,params.wV,params.wG,params.wF);
    
        log_a =    ( sample_H-propos_H ) ...
              +    ( sample_c-propos_c ) ...
              + beta*(propos_L-sample_L) ...
              + (params.h_prior_phi-1)*log(propos_h/sample_h) + params.h_prior_phi*(sample_h-propos_h)/params.h_prior_ref ...
              + (params.C_prior_phi-1)*log(propos_C/sample_C) + params.C_prior_phi*(sample_C-propos_C)/params.C_prior_ref ;
    
        % carry acc test
        if U_prop < log_a
            sample_h = propos_h;
            sample_C = propos_C;
            sample_L = propos_L;

            rec(1) = rec(1) + 1;
            break % while true
        else
            % update intervals
            if T < 0
                T_min = T;
            else
                T_max = T;
            end
            % prepare proposal
            T = T_min + (T_max-T_min)*rand;
        end % acc

    end % while true

end % rep

