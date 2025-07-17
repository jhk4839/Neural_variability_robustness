classdef Index < types.hdmf_common.Data & types.untyped.DatasetClass
% INDEX - Pointers that index data values.
%
% Required Properties:
%  data


% REQUIRED PROPERTIES
properties
    target; % REQUIRED (Object reference to Data) Target dataset that this index applies to.
end

methods
    function obj = Index(varargin)
        % INDEX - Constructor for Index
        %
        % Syntax:
        %  index = types.hdmf_common.INDEX() creates a Index object with unset property values.
        %
        %  index = types.hdmf_common.INDEX(Name, Value) creates a Index object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - data (any) - No description
        %
        %  - target (Object reference to Data) - Target dataset that this index applies to.
        %
        % Output Arguments:
        %  - index (types.hdmf_common.Index) - A Index object
        
        obj = obj@types.hdmf_common.Data(varargin{:});
        
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        addParameter(p, 'data',[]);
        addParameter(p, 'target',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.data = p.Results.data;
        obj.target = p.Results.target;
        if strcmp(class(obj), 'types.hdmf_common.Index')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
    end
    %% SETTERS
    function set.target(obj, val)
        obj.target = obj.validate_target(val);
    end
    %% VALIDATORS
    
    function val = validate_data(obj, val)
    end
    function val = validate_target(obj, val)
        % Reference to type `Data`
        val = types.util.checkDtype('target', 'types.untyped.ObjectView', val);
        if isa(val, 'types.untyped.DataStub')
            if 1 == val.ndims
                valsz = [val.dims 1];
            else
                valsz = val.dims;
            end
        elseif istable(val)
            valsz = [height(val) 1];
        elseif ischar(val)
            valsz = [size(val, 1) 1];
        else
            valsz = size(val);
        end
        validshapes = {[1]};
        types.util.checkDims(valsz, validshapes);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.hdmf_common.Data(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        io.writeAttribute(fid, [fullpath '/target'], obj.target);
    end
end

end