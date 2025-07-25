classdef DynamicTable < types.hdmf_common.Container & types.untyped.GroupClass
% DYNAMICTABLE - A group containing multiple datasets that are aligned on the first dimension (Currently, this requirement if left up to APIs to check and enforce). Apart from a column that contains unique identifiers for each row there are no other required datasets. Users are free to add any number of VectorData objects here. Table functionality is already supported through compound types, which is analogous to storing an array-of-structs. DynamicTable can be thought of as a struct-of-arrays. This provides an alternative structure to choose from when optimizing storage for anticipated access patterns. Additionally, this type provides a way of creating a table without having to define a compound type up front. Although this convenience may be attractive, users should think carefully about how data will be accessed. DynamicTable is more appropriate for column-centric access, whereas a dataset with a compound type would be more appropriate for row-centric access. Finally, data size should also be taken into account. For small tables, performance loss may be an acceptable trade-off for the flexibility of a DynamicTable. For example, DynamicTable was originally developed for storing trial data and spike unit metadata. Both of these use cases are expected to produce relatively small tables, so the spatial locality of multiple datasets present in a DynamicTable is not expected to have a significant performance impact. Additionally, requirements of trial and unit metadata tables are sufficiently diverse that performance implications can be overlooked in favor of usability.
%
% Required Properties:
%  id


% REQUIRED PROPERTIES
properties
    colnames; % REQUIRED (char) The names of the columns in this table. This should be used to specify an order to the columns.
    description; % REQUIRED (char) Description of what is in this dynamic table.
    id; % REQUIRED (ElementIdentifiers) Array of unique identifiers for the rows of this dynamic table.
end
% OPTIONAL PROPERTIES
properties
    vectordata; %  (VectorData) Vector columns of this dynamic table.
    vectorindex; %  (VectorIndex) Indices for the vector columns of this dynamic table.
end

methods
    function obj = DynamicTable(varargin)
        % DYNAMICTABLE - Constructor for DynamicTable
        %
        % Syntax:
        %  dynamicTable = types.hdmf_common.DYNAMICTABLE() creates a DynamicTable object with unset property values.
        %
        %  dynamicTable = types.hdmf_common.DYNAMICTABLE(Name, Value) creates a DynamicTable object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - colnames (char) - The names of the columns in this table. This should be used to specify an order to the columns.
        %
        %  - description (char) - Description of what is in this dynamic table.
        %
        %  - id (ElementIdentifiers) - Array of unique identifiers for the rows of this dynamic table.
        %
        %  - vectordata (VectorData) - Vector columns of this dynamic table.
        %
        %  - vectorindex (VectorIndex) - Indices for the vector columns of this dynamic table.
        %
        % Output Arguments:
        %  - dynamicTable (types.hdmf_common.DynamicTable) - A DynamicTable object
        
        obj = obj@types.hdmf_common.Container(varargin{:});
        [obj.vectordata, ivarargin] = types.util.parseConstrained(obj,'vectordata', 'types.hdmf_common.VectorData', varargin{:});
        varargin(ivarargin) = [];
        [obj.vectorindex, ivarargin] = types.util.parseConstrained(obj,'vectorindex', 'types.hdmf_common.VectorIndex', varargin{:});
        varargin(ivarargin) = [];
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        addParameter(p, 'colnames',[]);
        addParameter(p, 'description',[]);
        addParameter(p, 'id',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.colnames = p.Results.colnames;
        obj.description = p.Results.description;
        obj.id = p.Results.id;
        if strcmp(class(obj), 'types.hdmf_common.DynamicTable')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
    end
    %% SETTERS
    function set.colnames(obj, val)
        obj.colnames = obj.validate_colnames(val);
    end
    function set.description(obj, val)
        obj.description = obj.validate_description(val);
    end
    function set.id(obj, val)
        obj.id = obj.validate_id(val);
    end
    function set.vectordata(obj, val)
        obj.vectordata = obj.validate_vectordata(val);
    end
    function set.vectorindex(obj, val)
        obj.vectorindex = obj.validate_vectorindex(val);
    end
    %% VALIDATORS
    
    function val = validate_colnames(obj, val)
        val = types.util.checkDtype('colnames', 'char', val);
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
    function val = validate_description(obj, val)
        val = types.util.checkDtype('description', 'char', val);
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
    function val = validate_id(obj, val)
        val = types.util.checkDtype('id', 'types.hdmf_common.ElementIdentifiers', val);
    end
    function val = validate_vectordata(obj, val)
        constrained = { 'types.hdmf_common.VectorData' };
        types.util.checkSet('vectordata', struct(), constrained, val);
    end
    function val = validate_vectorindex(obj, val)
        constrained = { 'types.hdmf_common.VectorIndex' };
        types.util.checkSet('vectorindex', struct(), constrained, val);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.hdmf_common.Container(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        io.writeAttribute(fid, [fullpath '/colnames'], obj.colnames, 'forceArray');
        io.writeAttribute(fid, [fullpath '/description'], obj.description);
        refs = obj.id.export(fid, [fullpath '/id'], refs);
        if ~isempty(obj.vectordata)
            refs = obj.vectordata.export(fid, fullpath, refs);
        end
        if ~isempty(obj.vectorindex)
            refs = obj.vectorindex.export(fid, fullpath, refs);
        end
    end
    %% TABLE METHODS
    function addRow(obj, varargin)
        types.util.dynamictable.addRow(obj, varargin{:});
    end
    
    function addColumn(obj, varargin)
        types.util.dynamictable.addColumn(obj, varargin{:});
    end
    
    function row = getRow(obj, id, varargin)
        row = types.util.dynamictable.getRow(obj, id, varargin{:});
    end
    
    function table = toTable(obj, varargin)
        table = types.util.dynamictable.nwbToTable(obj, varargin{:});
    end
    
    function clear(obj)
        types.util.dynamictable.clear(obj);
    end
end

end