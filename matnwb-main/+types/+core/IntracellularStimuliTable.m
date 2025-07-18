classdef IntracellularStimuliTable < types.hdmf_common.DynamicTable & types.untyped.GroupClass
% INTRACELLULARSTIMULITABLE - Table for storing intracellular stimulus related metadata.
%
% Required Properties:
%  id, stimulus


% REQUIRED PROPERTIES
properties
    stimulus; % REQUIRED (TimeSeriesReferenceVectorData) Column storing the reference to the recorded stimulus for the recording (rows).
end

methods
    function obj = IntracellularStimuliTable(varargin)
        % INTRACELLULARSTIMULITABLE - Constructor for IntracellularStimuliTable
        %
        % Syntax:
        %  intracellularStimuliTable = types.core.INTRACELLULARSTIMULITABLE() creates a IntracellularStimuliTable object with unset property values.
        %
        %  intracellularStimuliTable = types.core.INTRACELLULARSTIMULITABLE(Name, Value) creates a IntracellularStimuliTable object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - colnames (char) - The names of the columns in this table. This should be used to specify an order to the columns.
        %
        %  - id (ElementIdentifiers) - Array of unique identifiers for the rows of this dynamic table.
        %
        %  - stimulus (TimeSeriesReferenceVectorData) - Column storing the reference to the recorded stimulus for the recording (rows).
        %
        %  - vectordata (VectorData) - Vector columns, including index columns, of this dynamic table.
        %
        % Output Arguments:
        %  - intracellularStimuliTable (types.core.IntracellularStimuliTable) - A IntracellularStimuliTable object
        
        varargin = [{'description' 'Table for storing intracellular stimulus related metadata.'} varargin];
        obj = obj@types.hdmf_common.DynamicTable(varargin{:});
        
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        addParameter(p, 'description',[]);
        addParameter(p, 'stimulus',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.description = p.Results.description;
        obj.stimulus = p.Results.stimulus;
        if strcmp(class(obj), 'types.core.IntracellularStimuliTable')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
        if strcmp(class(obj), 'types.core.IntracellularStimuliTable')
            types.util.dynamictable.checkConfig(obj);
        end
    end
    %% SETTERS
    function set.stimulus(obj, val)
        obj.stimulus = obj.validate_stimulus(val);
    end
    %% VALIDATORS
    
    function val = validate_description(obj, val)
        if isequal(val, 'Table for storing intracellular stimulus related metadata.')
            val = 'Table for storing intracellular stimulus related metadata.';
        else
            error('NWB:Type:ReadOnlyProperty', 'Unable to set the ''description'' property of class ''<a href="matlab:doc types.core.IntracellularStimuliTable">IntracellularStimuliTable</a>'' because it is read-only.')
        end
    end
    function val = validate_stimulus(obj, val)
        val = types.util.checkDtype('stimulus', 'types.core.TimeSeriesReferenceVectorData', val);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.hdmf_common.DynamicTable(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        refs = obj.stimulus.export(fid, [fullpath '/stimulus'], refs);
    end
end

end