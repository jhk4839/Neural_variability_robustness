classdef LFP < types.core.NWBDataInterface & types.untyped.GroupClass
% LFP - LFP data from one or more channels. The electrode map in each published ElectricalSeries will identify which channels are providing LFP data. Filter properties should be noted in the ElectricalSeries description or comments field.
%
% Required Properties:
%  electricalseries


% REQUIRED PROPERTIES
properties
    electricalseries; % REQUIRED (ElectricalSeries) ElectricalSeries object(s) containing LFP data for one or more channels.
end

methods
    function obj = LFP(varargin)
        % LFP - Constructor for LFP
        %
        % Syntax:
        %  lFP = types.core.LFP() creates a LFP object with unset property values.
        %
        %  lFP = types.core.LFP(Name, Value) creates a LFP object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - electricalseries (ElectricalSeries) - ElectricalSeries object(s) containing LFP data for one or more channels.
        %
        % Output Arguments:
        %  - lFP (types.core.LFP) - A LFP object
        
        obj = obj@types.core.NWBDataInterface(varargin{:});
        [obj.electricalseries, ivarargin] = types.util.parseConstrained(obj,'electricalseries', 'types.core.ElectricalSeries', varargin{:});
        varargin(ivarargin) = [];
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        misc.parseSkipInvalidName(p, varargin);
        if strcmp(class(obj), 'types.core.LFP')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
    end
    %% SETTERS
    function set.electricalseries(obj, val)
        obj.electricalseries = obj.validate_electricalseries(val);
    end
    %% VALIDATORS
    
    function val = validate_electricalseries(obj, val)
        namedprops = struct();
        constrained = {'types.core.ElectricalSeries'};
        types.util.checkSet('electricalseries', namedprops, constrained, val);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.core.NWBDataInterface(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        refs = obj.electricalseries.export(fid, fullpath, refs);
    end
end

end