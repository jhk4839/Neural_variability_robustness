classdef Units < types.hdmf_common.DynamicTable & types.untyped.GroupClass
% UNITS - Data about spiking units. Event times of observed units (e.g. cell, synapse, etc.) should be concatenated and stored in spike_times.
%
% Required Properties:
%  id


% OPTIONAL PROPERTIES
properties
    electrode_group; %  (VectorData) Electrode group that each spike unit came from.
    electrodes; %  (DynamicTableRegion) Electrode that each spike unit came from, specified using a DynamicTableRegion.
    electrodes_index; %  (VectorIndex) Index into electrodes.
    obs_intervals; %  (VectorData) Observation intervals for each unit.
    obs_intervals_index; %  (VectorIndex) Index into the obs_intervals dataset.
    spike_times; %  (VectorData) Spike times for each unit.
    spike_times_index; %  (VectorIndex) Index into the spike_times dataset.
    waveform_mean; %  (VectorData) Spike waveform mean for each spike unit.
    waveform_sd; %  (VectorData) Spike waveform standard deviation for each spike unit.
end

methods
    function obj = Units(varargin)
        % UNITS - Constructor for Units
        %
        % Syntax:
        %  units = types.core.UNITS() creates a Units object with unset property values.
        %
        %  units = types.core.UNITS(Name, Value) creates a Units object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - colnames (char) - The names of the columns in this table. This should be used to specify an order to the columns.
        %
        %  - description (char) - Description of what is in this dynamic table.
        %
        %  - electrode_group (VectorData) - Electrode group that each spike unit came from.
        %
        %  - electrodes (DynamicTableRegion) - Electrode that each spike unit came from, specified using a DynamicTableRegion.
        %
        %  - electrodes_index (VectorIndex) - Index into electrodes.
        %
        %  - id (ElementIdentifiers) - Array of unique identifiers for the rows of this dynamic table.
        %
        %  - obs_intervals (VectorData) - Observation intervals for each unit.
        %
        %  - obs_intervals_index (VectorIndex) - Index into the obs_intervals dataset.
        %
        %  - spike_times (VectorData) - Spike times for each unit.
        %
        %  - spike_times_index (VectorIndex) - Index into the spike_times dataset.
        %
        %  - vectordata (VectorData) - Vector columns of this dynamic table.
        %
        %  - vectorindex (VectorIndex) - Indices for the vector columns of this dynamic table.
        %
        %  - waveform_mean (VectorData) - Spike waveform mean for each spike unit.
        %
        %  - waveform_sd (VectorData) - Spike waveform standard deviation for each spike unit.
        %
        % Output Arguments:
        %  - units (types.core.Units) - A Units object
        
        obj = obj@types.hdmf_common.DynamicTable(varargin{:});
        
        
        p = inputParser;
        p.KeepUnmatched = true;
        p.PartialMatching = false;
        p.StructExpand = false;
        addParameter(p, 'colnames',[]);
        addParameter(p, 'description',[]);
        addParameter(p, 'electrode_group',[]);
        addParameter(p, 'electrodes',[]);
        addParameter(p, 'electrodes_index',[]);
        addParameter(p, 'obs_intervals',[]);
        addParameter(p, 'obs_intervals_index',[]);
        addParameter(p, 'spike_times',[]);
        addParameter(p, 'spike_times_index',[]);
        addParameter(p, 'waveform_mean',[]);
        addParameter(p, 'waveform_sd',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.colnames = p.Results.colnames;
        obj.description = p.Results.description;
        obj.electrode_group = p.Results.electrode_group;
        obj.electrodes = p.Results.electrodes;
        obj.electrodes_index = p.Results.electrodes_index;
        obj.obs_intervals = p.Results.obs_intervals;
        obj.obs_intervals_index = p.Results.obs_intervals_index;
        obj.spike_times = p.Results.spike_times;
        obj.spike_times_index = p.Results.spike_times_index;
        obj.waveform_mean = p.Results.waveform_mean;
        obj.waveform_sd = p.Results.waveform_sd;
        if strcmp(class(obj), 'types.core.Units')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
        if strcmp(class(obj), 'types.core.Units')
            types.util.dynamictable.checkConfig(obj);
        end
    end
    %% SETTERS
    function set.electrode_group(obj, val)
        obj.electrode_group = obj.validate_electrode_group(val);
    end
    function set.electrodes(obj, val)
        obj.electrodes = obj.validate_electrodes(val);
    end
    function set.electrodes_index(obj, val)
        obj.electrodes_index = obj.validate_electrodes_index(val);
    end
    function set.obs_intervals(obj, val)
        obj.obs_intervals = obj.validate_obs_intervals(val);
    end
    function set.obs_intervals_index(obj, val)
        obj.obs_intervals_index = obj.validate_obs_intervals_index(val);
    end
    function set.spike_times(obj, val)
        obj.spike_times = obj.validate_spike_times(val);
    end
    function set.spike_times_index(obj, val)
        obj.spike_times_index = obj.validate_spike_times_index(val);
    end
    function set.waveform_mean(obj, val)
        obj.waveform_mean = obj.validate_waveform_mean(val);
    end
    function set.waveform_sd(obj, val)
        obj.waveform_sd = obj.validate_waveform_sd(val);
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
    function val = validate_electrode_group(obj, val)
        val = types.util.checkDtype('electrode_group', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_electrodes(obj, val)
        val = types.util.checkDtype('electrodes', 'types.hdmf_common.DynamicTableRegion', val);
    end
    function val = validate_electrodes_index(obj, val)
        val = types.util.checkDtype('electrodes_index', 'types.hdmf_common.VectorIndex', val);
    end
    function val = validate_obs_intervals(obj, val)
        val = types.util.checkDtype('obs_intervals', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_obs_intervals_index(obj, val)
        val = types.util.checkDtype('obs_intervals_index', 'types.hdmf_common.VectorIndex', val);
    end
    function val = validate_spike_times(obj, val)
        val = types.util.checkDtype('spike_times', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_spike_times_index(obj, val)
        val = types.util.checkDtype('spike_times_index', 'types.hdmf_common.VectorIndex', val);
    end
    function val = validate_waveform_mean(obj, val)
        val = types.util.checkDtype('waveform_mean', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_waveform_sd(obj, val)
        val = types.util.checkDtype('waveform_sd', 'types.hdmf_common.VectorData', val);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.hdmf_common.DynamicTable(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        if ~isempty(obj.electrode_group)
            refs = obj.electrode_group.export(fid, [fullpath '/electrode_group'], refs);
        end
        if ~isempty(obj.electrodes)
            refs = obj.electrodes.export(fid, [fullpath '/electrodes'], refs);
        end
        if ~isempty(obj.electrodes_index)
            refs = obj.electrodes_index.export(fid, [fullpath '/electrodes_index'], refs);
        end
        if ~isempty(obj.obs_intervals)
            refs = obj.obs_intervals.export(fid, [fullpath '/obs_intervals'], refs);
        end
        if ~isempty(obj.obs_intervals_index)
            refs = obj.obs_intervals_index.export(fid, [fullpath '/obs_intervals_index'], refs);
        end
        if ~isempty(obj.spike_times)
            refs = obj.spike_times.export(fid, [fullpath '/spike_times'], refs);
        end
        if ~isempty(obj.spike_times_index)
            refs = obj.spike_times_index.export(fid, [fullpath '/spike_times_index'], refs);
        end
        if ~isempty(obj.waveform_mean)
            refs = obj.waveform_mean.export(fid, [fullpath '/waveform_mean'], refs);
        end
        if ~isempty(obj.waveform_sd)
            refs = obj.waveform_sd.export(fid, [fullpath '/waveform_sd'], refs);
        end
    end
end

end