function log_L = get_log_like(u_cnt,dW_cnt,wV,wG,wF)
% M_cnt = wM +    wG  *u_cnt;
% V_cnt = wV + wF*wG^2*u_cnt;

V_cnt = wV/wG^2 + wF * u_cnt;
log_L = - 0.5 * sum( log(V_cnt) + (dW_cnt-u_cnt).^2 ./ V_cnt , 'all' );
