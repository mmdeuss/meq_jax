function V = strip_function_handles(V,method)
% Delete function handles to make it loadable in Octave.

doremove = strcmp(method,'remove');
isfh = @(v) isa(v,'function_handle');

if isstruct(V)
  for fld = fieldnames(V)'
    fieldval = V.(fld{:});
    ishandle = isfh(fieldval);
    iscellwithhandle = iscell(fieldval) && any(cellfun(@(x) isfh(x),fieldval(:)));
    if doremove && (ishandle || iscellwithhandle)
      V = rmfield(V,fld{:});
    else
      V.(fld{:}) = strip_function_handles(fieldval,method);
    end
  end
elseif iscell(V)
  for ii = 1:numel(V)
    V{ii} = strip_function_handles(V{ii},method);
  end
elseif isa(V,'function_handle')
  V = functions(V); % replace by struct with information from functions()
  switch method
    case 'string'
      V = V.function; % keep function name only
    case 'functions' 
      % keep output of 'functions()' call on the handle
    case {'empty', 'remove'}
      % return empty
      V = [];
    otherwise, error('unknown method %s',method);
  end
end

end