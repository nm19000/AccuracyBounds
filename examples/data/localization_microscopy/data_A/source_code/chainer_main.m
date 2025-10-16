function chain = chainer_main(chain_init,d_length,opts,flag_status,flag_visual)
% to init:
% chain = chainer_main([]   ,  0, opts, true, []  );
% to expand:
% chain = chainer_main(chain,+25, []  , true, true);
% to reduce:
% chain = chainer_main(chain,-10, []  , true, []  );
% to reset record:
% chain = chainer_main(chain, [], []  , true, []  );

tic_id = tic;


% initialize the seed or use seed for last expansion
if isempty(chain_init)
    rng('shuffle');
else
    rng(chain_init.random);
end


% init chain --------------------------------------------------------------
if d_length == 0

    % MCMC
    chain.params = chainer_init_params(opts);
    chain.length = 1;
    chain.stride = 1;
    chain.ledger = nan(0,2); % total wall time without Initialization
    chain.sizeGB = nan;      % current chain memory size
    chain.record = [];       % acceptance rates
    chain.sample = [];

    chain.sample = chainer_init_sample(chain.params,opts);
    
    % history
    chain.i = cast( chain.sample.i, 'uint64' );
    chain.P = cast( chain.sample.P, 'double' );
    chain.L = cast( chain.sample.L, 'double' );
    chain.x = cast( chain.sample.x, 'double' );
    chain.y = cast( chain.sample.y, 'double' );
    chain.h = cast( chain.sample.h, 'single' );
    chain.C = cast( chain.sample.C, 'single' );
    
    if flag_status
        disp('CHAINER: chain initiated')
    end

    

% expand chain ------------------------------------------------------------
elseif d_length > 0

    chain.params = chain_init.params;
    chain.length = chain_init.length + d_length;
    chain.stride = chain_init.stride;
    chain.ledger = chain_init.ledger;
    chain.sizeGB = nan;
    chain.record = chain_init.record;
    chain.sample = chain_init.sample;
    
    chain.i = cat( 2 , chain_init.i , zeros( 1              , d_length , 'like',chain_init.i ) );
    chain.P = cat( 2 , chain_init.P ,   nan( get_log_probs  , d_length , 'like',chain_init.P ) );
    chain.L = cat( 2 , chain_init.L ,   nan( 1              , d_length , 'like',chain_init.L ) );
    chain.x = cat( 2 , chain_init.x ,   nan( 1              , d_length , 'like',chain_init.x ) );
    chain.y = cat( 2 , chain_init.y ,   nan( 1              , d_length , 'like',chain_init.y ) );
    chain.h = cat( 2 , chain_init.h ,   nan( 1              , d_length , 'like',chain_init.h ) );
    chain.C = cat( 2 , chain_init.C ,   nan( 1              , d_length , 'like',chain_init.C ) );
    
    if flag_visual
        Gim = chainer_visualize([],chain);
    end
    
    %---------------------------- expand chain
    r = chain_init.length+1;
    while r <= chain.length
        
        chain.sample = sampler_update(chain.sample,chain.params);
        
        [ chain.params, chain.record, chain.sample.rec ] = sampler_adapt_proposals(chain.params,chain.record,chain.sample.rec,chain.sample.i);
        
        if mod(chain.sample.i,chain.stride) == 0
            
            chain.i(  r) = chain.sample.i;
            chain.P(:,r) = chain.sample.P;
            chain.L(  r) = chain.sample.L;
            chain.x(  r) = chain.sample.x;
            chain.y(  r) = chain.sample.y;
            chain.h(  r) = chain.sample.h;
            chain.C(  r) = chain.sample.C;
            
            if flag_visual
                chainer_visualize(Gim,chain);
            end
            
            if flag_status
                disp([  'i = ', num2str(chain.sample.i,'%d'), ...
                     ' - acc = ', ...
                                num2str( chain.sample.rec(1,:)./chain.sample.rec(2,:)  * 100 ,'%#6.2f') , ' %', ...
                     ])
            end
            
            r = r+1;
        end
    end    

    if flag_status
        disp('CHAINER: chain expanded')
    end


% reduce chain ------------------------------------------------------------
elseif d_length < 0

    d_length = min(-d_length,chain_init.length);

    chain.params = chain_init.params;
    chain.length = d_length;
    chain.stride = nan;
    chain.ledger = chain_init.ledger;
    chain.sizeGB = nan;
    chain.record = chain_init.record;
    chain.sample = chain_init.sample;
    
    ind = mod(chain_init.length,d_length)+(floor(chain_init.length/d_length)*(1:d_length));

    chain.i = chain_init.i(  ind);
    chain.P = chain_init.P(:,ind);
    chain.L = chain_init.L(  ind);
    chain.x = chain_init.x(  ind);
    chain.y = chain_init.y(  ind);
    chain.h = chain_init.h(  ind);
    chain.C = chain_init.C(  ind);
    
    chain.stride = double(chain.i(2)-chain.i(1));
    
    if flag_status
        disp('CHAINER: chain reduced')
    end
    
    
% reset chain -------------------------------------------------------------
elseif isempty(d_length)

    chain = chain_init;
    
    chain.record = [chain.record; [chain.sample.i,chain.sample.rec(1,:)./chain.sample.rec(2,:)] ];

    chain.sample.rec(1,:) = 0;
    chain.sample.rec(2,:) = realmin;
    chain.sample.rec(3,:) = realmin;

    if flag_status
        disp('CHAINER: chain reset')
    end
    
end


% store the seed for future expansion
chain.random = rng();



%% book-keeping
chain.sizeGB = get_sizeGB(chain);               % mem size

% ledger
wall_time = toc(tic_id);
chain.ledger = [chain.ledger; double(chain.i(end)), wall_time];

if flag_status
    disp(['( wall time = ',num2str(wall_time),' s, overall wall time = ',num2str(sum(chain.ledger(:,2))),' s )'])
end



end





%% auxiliary functions

function sizeGB = get_sizeGB(chain)
    sizeGB = whos( inputname(1) );
    sizeGB = sizeGB.bytes/1024^3;
end


function str = get_int_type(L)
    if     L <= intmax('uint8')
        str = 'uint8';
    elseif L <= intmax('uint16')
        str = 'uint16';
    elseif L <= intmax('uint32')
        str = 'uint32';
    elseif L <= intmax('uint64')
        str = 'uint64';
    else
        error('L way too large !!!');
    end
end
    








%% adapter
function [ params, record, rec ] = sampler_adapt_proposals(params,record,rec,i)

rep = 25;

if i <= rep*20

    if mod(i,rep)==0
%         disp('=== ADAPTER ===')
%         acc_old = rec(1,[2 4])./rec(2,[2 4]);
%         record = [record; [i,acc_old] ];
%         r = min( max( acc_old/0.75 , 0.1 ), 10 );
%         if rec(2,[2 4])<1
%         	r = 1;
%         end
%         disp(['acc = ',num2str( acc_old                                ,2 )])
%         disp(['old = ',num2str( [params.propos_h_pm params.propos_h_cm],2 )])
%         params.propos_h_pm = params.propos_h_pm * r(1);
%         params.propos_h_cm = params.propos_h_cm * r(2);
%         disp(['new = ',num2str( [params.propos_h_pm params.propos_h_cm],2 )])
%         disp(['  r = ',num2str( r                                      ,2 )])
%         disp('=== === === ===')
%         rec(1,:) = 0;
%         rec(2,:) = realmin;
    end
    
end

end


