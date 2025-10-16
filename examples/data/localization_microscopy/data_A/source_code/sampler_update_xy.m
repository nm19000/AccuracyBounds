function [sample_x,sample_y,sample_L,rec] = sampler_update_xy( ...
          sample_x,sample_y,sample_L,rec, h,C,params,beta)


V_cnt = params.t_exp*C*(params.gx_max-params.gx_min)/params.Px ...
                      *(params.gy_max-params.gy_min)/params.Py ;

%% update active
for rep = 1:sum(rand(1*params.rep_xy*2,1)<0.5)

    % prepare sampler
    sample_X = params.sig_X*randn;
    tempor_X = params.sig_X*randn;
    sample_Y = params.sig_Y*randn;
    tempor_Y = params.sig_Y*randn;
    
    % pick slice
    U_prop = log(rand);

    % pick interval
    T = 2*pi*rand;
    T_min = T - 2*pi;
    T_max = T;
    
    while true 

        rec(2) = rec(2) + 1;

        propos_X = cos(T)*sample_X + sin(T)*tempor_X;
        propos_Y = cos(T)*sample_Y + sin(T)*tempor_Y;

        propos_x = sample_x + ( sample_X - propos_X );
        propos_y = sample_y + ( sample_Y - propos_Y );

        if propos_x<params.x_prior_min || propos_x>params.x_prior_max ...
        || propos_y<params.y_prior_min || propos_y>params.y_prior_max
            log_a = -inf;
        else
            propos_L = get_log_like(V_cnt+params.t_exp*h*img_get_PSF(propos_x,propos_y,...
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
                                                                     params.f_nom,'r'), ...
                                                                     params.dW_cnt,params.wV,params.wG,params.wF);
            log_a = beta*(propos_L-sample_L);
        end % log_a

        % carry acc test
        if U_prop < log_a
            sample_x = propos_x;
            sample_y = propos_y;
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

