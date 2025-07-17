classdef VectorIndex < types.hdmf_common.Index & types.untyped.DatasetClass
% VECTORINDEX - Used with VectorData to encode a ragged array. An array of indices into the first dimension of the target VectorData, and forming a map between the rows of a DynamicTable and the indices of the VectorData.
%
% Required Properties:
%  data



methods
    function obj = VectorIndex(varargin)
        % VECTORINDEX - Constructor for VectorIndex
        %
        % Syntax:
        %  vectorIndex = types.hdmf_common.VECTORINDEX() creates a VectorIndex object with unset property values.
        %
        %  vectorIndex = types.hdmf_common.VECTORINDEX(Name, Value) creates a VectorIndex object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - data (any) - No description
        %
        %  - target (Object reference to VectorData) - Reference to the target dataset that this index applies to.
        %
        % Output Arguments:
        %  - vectorIndex (types.hdmf_common.VectorIndex) - A VectorIndex object
        
        obj = obj@types.hdmf_common.Index(varargin{:});
        
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        addParameter(p, 'data',[]);
        addParameter(p, 'target',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.data = p.Results.data;
        obj.target = p.Results.target;
        if strcmp(class(obj), 'types.hdmf_common.VectorIndex')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
    end
    %% SETTERS
    
    %% VALIDATORS
    
    function val = validate_data(obj, val)
    end
    function val = validate_target(obj, val)
        % Reference to type `VectorData`
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
        refs = export@types.hdmf_common.Index(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
    end
end

end