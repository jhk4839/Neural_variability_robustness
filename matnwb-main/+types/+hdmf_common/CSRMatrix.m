classdef CSRMatrix < types.untyped.MetaClass & types.untyped.GroupClass
% CSRMATRIX - a compressed sparse row matrix
%
% Required Properties:
%  data, indices, indptr


% REQUIRED PROPERTIES
properties
    data; % REQUIRED (any) values in the matrix
    indices; % REQUIRED (int8) column indices
    indptr; % REQUIRED (int8) index pointer
    shape; % REQUIRED (int8) the shape of this sparse matrix
end

methods
    function obj = CSRMatrix(varargin)
        % CSRMATRIX - Constructor for CSRMatrix
        %
        % Syntax:
        %  cSRMatrix = types.hdmf_common.CSRMATRIX() creates a CSRMatrix object with unset property values.
        %
        %  cSRMatrix = types.hdmf_common.CSRMATRIX(Name, Value) creates a CSRMatrix object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - data (any) - values in the matrix
        %
        %  - indices (int8) - column indices
        %
        %  - indptr (int8) - index pointer
        %
        %  - shape (int8) - the shape of this sparse matrix
        %
        % Output Arguments:
        %  - cSRMatrix (types.hdmf_common.CSRMatrix) - A CSRMatrix object
        
        obj = obj@types.untyped.MetaClass(varargin{:});
        
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        addParameter(p, 'data',[]);
        addParameter(p, 'indices',[]);
        addParameter(p, 'indptr',[]);
        addParameter(p, 'shape',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.data = p.Results.data;
        obj.indices = p.Results.indices;
        obj.indptr = p.Results.indptr;
        obj.shape = p.Results.shape;
        if strcmp(class(obj), 'types.hdmf_common.CSRMatrix')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
    end
    %% SETTERS
    function set.data(obj, val)
        obj.data = obj.validate_data(val);
    end
    function set.indices(obj, val)
        obj.indices = obj.validate_indices(val);
    end
    function set.indptr(obj, val)
        obj.indptr = obj.validate_indptr(val);
    end
    function set.shape(obj, val)
        obj.shape = obj.validate_shape(val);
    end
    %% VALIDATORS
    
    function val = validate_data(obj, val)
    
    end
    function val = validate_indices(obj, val)
        val = types.util.checkDtype('indices', 'int8', val);
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
        validshapes = {[Inf]};
        types.util.checkDims(valsz, validshapes);
    end
    function val = validate_indptr(obj, val)
        val = types.util.checkDtype('indptr', 'int8', val);
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
        validshapes = {[Inf]};
        types.util.checkDims(valsz, validshapes);
    end
    function val = validate_shape(obj, val)
        val = types.util.checkDtype('shape', 'int8', val);
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
        validshapes = {[2]};
        types.util.checkDims(valsz, validshapes);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.untyped.MetaClass(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        if startsWith(class(obj.data), 'types.untyped.')
            refs = obj.data.export(fid, [fullpath '/data'], refs);
        elseif ~isempty(obj.data)
            io.writeDataset(fid, [fullpath '/data'], obj.data, 'forceArray');
        end
        if startsWith(class(obj.indices), 'types.untyped.')
            refs = obj.indices.export(fid, [fullpath '/indices'], refs);
        elseif ~isempty(obj.indices)
            io.writeDataset(fid, [fullpath '/indices'], obj.indices, 'forceArray');
        end
        if startsWith(class(obj.indptr), 'types.untyped.')
            refs = obj.indptr.export(fid, [fullpath '/indptr'], refs);
        elseif ~isempty(obj.indptr)
            io.writeDataset(fid, [fullpath '/indptr'], obj.indptr, 'forceArray');
        end
        io.writeAttribute(fid, [fullpath '/shape'], obj.shape, 'forceArray');
    end
end

end