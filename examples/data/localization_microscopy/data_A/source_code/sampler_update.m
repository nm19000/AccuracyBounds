function sample = sampler_update( sample, params )

%% update counter
sample.i = sample.i + 1;

%% random sweep
for tag = randperm(2)
    switch tag
        case 1
            [sample.x,sample.y,sample.L,sample.rec(:,tag)] = sampler_update_xy( ...
             sample.x,sample.y,sample.L,sample.rec(:,tag), sample.h,sample.C,params,1);
        case 2
            [sample.h,sample.C,sample.L,sample.rec(:,tag)] = sampler_update_hC( ...
             sample.h,sample.C,sample.L,sample.rec(:,tag), sample.x,sample.y,params,1);
        otherwise
            error('Unknown sampler requested!')
    end
end

%% book-keeping
sample.P = get_log_probs(sample.x,sample.y,sample.h,sample.C,params);

