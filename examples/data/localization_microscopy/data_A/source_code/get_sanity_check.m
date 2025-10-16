function ch = get_sanity_check(log_a)
% returns true when the checks are successfully passed 
% returns false otherwise

% use like this
% if ~get_sanity_check(log_a)
%     keyboard
% end

ch =   isscalar(log_a) ...
 &&   ~isnan(log_a) ...
 &&   isreal(log_a) ;
