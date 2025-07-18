classdef SweepTable < types.hdmf_common.DynamicTable & types.untyped.GroupClass
% SWEEPTABLE - The table which groups different PatchClampSeries together.
%
% Required Properties:
%  id, series, series_index, sweep_number


% REQUIRED PROPERTIES
properties
    series; % REQUIRED (VectorData) The PatchClampSeries with the sweep number in that row.
    series_index; % REQUIRED (VectorIndex) Index for series.
    sweep_number; % REQUIRED (VectorData) Sweep number of the PatchClampSeries in that row.
end

methods
    function obj = SweepTable(varargin)
        % SWEEPTABLE - Constructor for SweepTable
        %
        % Syntax:
        %  sweepTable = types.core.SWEEPTABLE() creates a SweepTable object with unset property values.
        %
        %  sweepTable = types.core.SWEEPTABLE(Name, Value) creates a SweepTable object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - colnames (char) - The names of the columns in this table. This should be used to specify an order to the columns.
        %
        %  - description (char) - Description of what is in this dynamic table.
        %
        %  - id (ElementIdentifiers) - Array of unique identifiers for the rows of this dynamic table.
        %
        %  - series (VectorData) - The PatchClampSeries with the sweep number in that row.
        %
        %  - series_index (VectorIndex) - Index for series.
        %
        %  - sweep_number (VectorData) - Sweep number of the PatchClampSeries in that row.
        %
        %  - vectordata (VectorData) - Vector columns of this dynamic table.
        %
        %  - vectorindex (VectorIndex) - Indices for the vector columns of this dynamic table.
        %
        % Output Arguments:
        %  - sweepTable (types.core.SweepTable) - A SweepTable object
        
        obj = obj@types.hdmf_common.DynamicTable(varargin{:});
        
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        addParameter(p, 'colnames',[]);
        addParameter(p, 'description',[]);
        addParameter(p, 'series',[]);
        addParameter(p, 'series_index',[]);
        addParameter(p, 'sweep_number',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.colnames = p.Results.colnames;
        obj.description = p.Results.description;
        obj.series = p.Results.series;
        obj.series_index = p.Results.series_index;
        obj.sweep_number = p.Results.sweep_number;
        if strcmp(class(obj), 'types.core.SweepTable')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
        if strcmp(class(obj), 'types.core.SweepTable')
            types.util.dynamictable.checkConfig(obj);
        end
    end
    %% SETTERS
    function set.series(obj, val)
        obj.series = obj.validate_series(val);
    end
    function set.series_index(obj, val)
        obj.series_index = obj.validate_series_index(val);
    end
    function set.sweep_number(obj, val)
        obj.sweep_number = obj.validate_sweep_number(val);
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
    function val = validate_series(obj, val)
        val = types.util.checkDtype('series', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_series_index(obj, val)
        val = types.util.checkDtype('series_index', 'types.hdmf_common.VectorIndex', val);
    end
    function val = validate_sweep_number(obj, val)
        val = types.util.checkDtype('sweep_number', 'types.hdmf_common.VectorData', val);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.hdmf_common.DynamicTable(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        refs = obj.series.export(fid, [fullpath '/series'], refs);
        refs = obj.series_index.export(fid, [fullpath '/series_index'], refs);
        refs = obj.sweep_number.export(fid, [fullpath '/sweep_number'], refs);
    end
end

end