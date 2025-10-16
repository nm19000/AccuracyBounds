function Gim = chainer_visualize(Gim,chain)

chain_P = sum(chain.P,1)+chain.L;

%% init
if isempty(Gim)

    num = 5;
    mum = 4;

    chain_i = double(chain.i(1)) + chain.stride*(0:chain.length-1)';
    i_lim = [max(chain_i(1),0.1*chain_i(end)) chain_i(end)+1];



    figure(10)
    set(gcf,'windowstyle','docked')
    clf

    tiledlayout(num,mum,'TileSpacing','compact')

    % --- Sample ----------------------------------------------------------

    ax = nexttile([num mum-1]);

    imagesc(ax,[chain.params.gy_min+0.5*chain.params.ds,chain.params.gy_max-0.5*chain.params.ds],...
               [chain.params.gx_min+0.5*chain.params.ds,chain.params.gx_max-0.5*chain.params.ds],...
               mean(chain.params.dW_cnt,3));

    axis(ax,'image')
    
    ax.XAxisLocation = 'top';

    c = colorbar(ax,'location','southoutside');
    c.Label.String = ['mean image value (',chain.params.units_image,')'];

    xlabel(ax,['y (',chain.params.units_length,')'])
    ylabel(ax,['x (',chain.params.units_length,')'])

    line(ax,[chain.params.y_prior_min chain.params.y_prior_max chain.params.y_prior_max chain.params.y_prior_min chain.params.y_prior_min],...
            [chain.params.x_prior_min chain.params.x_prior_min chain.params.x_prior_max chain.params.x_prior_max chain.params.x_prior_min],'color','m')

    title(ax,'MCMC sample')

    Gim.X = yline(chain.sample.x,'w');
    Gim.Y = xline(chain.sample.y,'w');

    
    % --- MCMC ------------------------------------------------------------
    ax_P = nexttile(  mum,[1 1]);
    ax_x = nexttile(2*mum,[1 1]);
    ax_y = nexttile(3*mum,[1 1]);
    ax_h = nexttile(4*mum,[1 1]);
    ax_C = nexttile(5*mum,[1 1]);
    
    ax_h.YScale = 'log';
    ax_C.YScale = 'log';

    ax_P.YAxisLocation = 'Right';
    ax_x.YAxisLocation = 'Right';
    ax_y.YAxisLocation = 'Right';
    ax_h.YAxisLocation = 'Right';
    ax_C.YAxisLocation = 'Right';
    
    ax_P.XLim = i_lim;
    ax_x.XLim = i_lim;
    ax_y.XLim = i_lim;
    ax_h.XLim = i_lim;
    ax_C.XLim = i_lim;
    
    title(ax_P,{'MCMC chain',['(stride=',num2str(chain.stride),')']})

    xlabel(ax_C, 'MCMC iteration (i)')

    ax_P.YGrid = 'on';
    ax_x.YGrid = 'on';
    ax_y.YGrid = 'on';
    ax_h.YGrid = 'on';
    ax_C.YGrid = 'on';

    ylabel(ax_P,{'logP_{post}';'(nat)'} )
    ylabel(ax_x,{'x',['(',chain.params.units_length,')']} )
    ylabel(ax_y,{'y',['(',chain.params.units_length,')']} )
    ylabel(ax_h,{'h',['(photons/',chain.params.units_time,')']} )
    ylabel(ax_C,{'C',['(photons/',chain.params.units_time,'/',chain.params.units_length,'^2)']} )

    if ~isempty( chain.ledger )
        xline(ax_P,chain.ledger(end,1));
        xline(ax_x,chain.ledger(end,1));
        xline(ax_y,chain.ledger(end,1));
        xline(ax_h,chain.ledger(end,1));
        xline(ax_C,chain.ledger(end,1));
    end

    yline(ax_x,chain.params.x_prior_min,'m')
    yline(ax_y,chain.params.y_prior_min,'m')

    yline(ax_x,chain.params.x_prior_max,'m')
    yline(ax_y,chain.params.y_prior_max,'m')

    Gim.P = line(ax_P,chain_i,chain_P,'marker','.');
    Gim.x = line(ax_x,chain_i,chain.x,'marker','.');
    Gim.y = line(ax_y,chain_i,chain.y,'marker','.');
    Gim.h = line(ax_h,chain_i,chain.h,'marker','.');
    Gim.C = line(ax_C,chain_i,chain.C,'marker','.');
    


    % --- ground ----------------------------------------------------------

    if isfield( chain.params,'ground' )
        yline(ax_x,chain.params.ground.x,'g')
        yline(ax_y,chain.params.ground.y,'g')
        yline(ax  ,chain.params.ground.x,'g')
        xline(ax  ,chain.params.ground.y,'g')
    end

end % init




Gim.X.Value = chain.sample.x;
Gim.Y.Value = chain.sample.y;

Gim.P.YData = chain_P;
Gim.x.YData = chain.x;
Gim.y.YData = chain.y;
Gim.h.YData = chain.h;
Gim.C.YData = chain.C;

drawnow



end % visualize
