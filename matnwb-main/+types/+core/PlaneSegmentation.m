classdef PlaneSegmentation < types.hdmf_common.DynamicTable & types.untyped.GroupClass
% PLANESEGMENTATION - Results from image segmentation of a specific imaging plane.
%
% Required Properties:
%  id


% REQUIRED PROPERTIES
properties
    imaging_plane; % REQUIRED ImagingPlane
end
% OPTIONAL PROPERTIES
properties
    image_mask; %  (VectorData) ROI masks for each ROI. Each image mask is the size of the original imaging plane (or volume) and members of the ROI are finite non-zero.
    pixel_mask; %  (VectorData) Pixel masks for each ROI: a list of indices and weights for the ROI. Pixel masks are concatenated and parsing of this dataset is maintained by the PlaneSegmentation
    pixel_mask_index; %  (VectorIndex) Index into pixel_mask.
    reference_images; %  (ImageSeries) One or more image stacks that the masks apply to (can be one-element stack).
    voxel_mask; %  (VectorData) Voxel masks for each ROI: a list of indices and weights for the ROI. Voxel masks are concatenated and parsing of this dataset is maintained by the PlaneSegmentation
    voxel_mask_index; %  (VectorIndex) Index into voxel_mask.
end

methods
    function obj = PlaneSegmentation(varargin)
        % PLANESEGMENTATION - Constructor for PlaneSegmentation
        %
        % Syntax:
        %  planeSegmentation = types.core.PLANESEGMENTATION() creates a PlaneSegmentation object with unset property values.
        %
        %  planeSegmentation = types.core.PLANESEGMENTATION(Name, Value) creates a PlaneSegmentation object where one or more property values are specified using name-value pairs.
        %
        % Input Arguments (Name-Value Arguments):
        %  - colnames (char) - The names of the columns in this table. This should be used to specify an order to the columns.
        %
        %  - description (char) - Description of what is in this dynamic table.
        %
        %  - id (ElementIdentifiers) - Array of unique identifiers for the rows of this dynamic table.
        %
        %  - image_mask (VectorData) - ROI masks for each ROI. Each image mask is the size of the original imaging plane (or volume) and members of the ROI are finite non-zero.
        %
        %  - imaging_plane (ImagingPlane) - Link to ImagingPlane object from which this data was generated.
        %
        %  - pixel_mask (VectorData) - Pixel masks for each ROI: a list of indices and weights for the ROI. Pixel masks are concatenated and parsing of this dataset is maintained by the PlaneSegmentation
        %
        %  - pixel_mask_index (VectorIndex) - Index into pixel_mask.
        %
        %  - reference_images (ImageSeries) - One or more image stacks that the masks apply to (can be one-element stack).
        %
        %  - vectordata (VectorData) - Vector columns of this dynamic table.
        %
        %  - vectorindex (VectorIndex) - Indices for the vector columns of this dynamic table.
        %
        %  - voxel_mask (VectorData) - Voxel masks for each ROI: a list of indices and weights for the ROI. Voxel masks are concatenated and parsing of this dataset is maintained by the PlaneSegmentation
        %
        %  - voxel_mask_index (VectorIndex) - Index into voxel_mask.
        %
        % Output Arguments:
        %  - planeSegmentation (types.core.PlaneSegmentation) - A PlaneSegmentation object
        
        obj = obj@types.hdmf_common.DynamicTable(varargin{:});
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
        addParameter(p, 'image_mask',[]);
        addParameter(p, 'imaging_plane',[]);
        addParameter(p, 'pixel_mask',[]);
        addParameter(p, 'pixel_mask_index',[]);
        addParameter(p, 'reference_images',types.untyped.Set());
        addParameter(p, 'voxel_mask',[]);
        addParameter(p, 'voxel_mask_index',[]);
        misc.parseSkipInvalidName(p, varargin);
        obj.colnames = p.Results.colnames;
        obj.description = p.Results.description;
        obj.id = p.Results.id;
        obj.image_mask = p.Results.image_mask;
        obj.imaging_plane = p.Results.imaging_plane;
        obj.pixel_mask = p.Results.pixel_mask;
        obj.pixel_mask_index = p.Results.pixel_mask_index;
        obj.reference_images = p.Results.reference_images;
        obj.voxel_mask = p.Results.voxel_mask;
        obj.voxel_mask_index = p.Results.voxel_mask_index;
        if strcmp(class(obj), 'types.core.PlaneSegmentation')
            cellStringArguments = convertContainedStringsToChars(varargin(1:2:end));
            types.util.checkUnset(obj, unique(cellStringArguments));
        end
        if strcmp(class(obj), 'types.core.PlaneSegmentation')
            types.util.dynamictable.checkConfig(obj);
        end
    end
    %% SETTERS
    function set.image_mask(obj, val)
        obj.image_mask = obj.validate_image_mask(val);
    end
    function set.imaging_plane(obj, val)
        obj.imaging_plane = obj.validate_imaging_plane(val);
    end
    function set.pixel_mask(obj, val)
        obj.pixel_mask = obj.validate_pixel_mask(val);
    end
    function set.pixel_mask_index(obj, val)
        obj.pixel_mask_index = obj.validate_pixel_mask_index(val);
    end
    function set.reference_images(obj, val)
        obj.reference_images = obj.validate_reference_images(val);
    end
    function set.voxel_mask(obj, val)
        obj.voxel_mask = obj.validate_voxel_mask(val);
    end
    function set.voxel_mask_index(obj, val)
        obj.voxel_mask_index = obj.validate_voxel_mask_index(val);
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
    function val = validate_image_mask(obj, val)
        val = types.util.checkDtype('image_mask', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_imaging_plane(obj, val)
        if isa(val, 'types.untyped.SoftLink')
            if isprop(val, 'target')
                types.util.checkDtype('imaging_plane', 'types.core.ImagingPlane', val.target);
            end
        else
            val = types.util.checkDtype('imaging_plane', 'types.core.ImagingPlane', val);
            if ~isempty(val)
                val = types.untyped.SoftLink(val);
            end
        end
    end
    function val = validate_pixel_mask(obj, val)
        val = types.util.checkDtype('pixel_mask', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_pixel_mask_index(obj, val)
        val = types.util.checkDtype('pixel_mask_index', 'types.hdmf_common.VectorIndex', val);
    end
    function val = validate_reference_images(obj, val)
        namedprops = struct();
        constrained = {'types.core.ImageSeries'};
        types.util.checkSet('reference_images', namedprops, constrained, val);
    end
    function val = validate_vectordata(obj, val)
        constrained = { 'types.hdmf_common.VectorData' };
        types.util.checkSet('vectordata', struct(), constrained, val);
    end
    function val = validate_vectorindex(obj, val)
        constrained = { 'types.hdmf_common.VectorIndex' };
        types.util.checkSet('vectorindex', struct(), constrained, val);
    end
    function val = validate_voxel_mask(obj, val)
        val = types.util.checkDtype('voxel_mask', 'types.hdmf_common.VectorData', val);
    end
    function val = validate_voxel_mask_index(obj, val)
        val = types.util.checkDtype('voxel_mask_index', 'types.hdmf_common.VectorIndex', val);
    end
    %% EXPORT
    function refs = export(obj, fid, fullpath, refs)
        refs = export@types.hdmf_common.DynamicTable(obj, fid, fullpath, refs);
        if any(strcmp(refs, fullpath))
            return;
        end
        if ~isempty(obj.image_mask)
            refs = obj.image_mask.export(fid, [fullpath '/image_mask'], refs);
        end
        refs = obj.imaging_plane.export(fid, [fullpath '/imaging_plane'], refs);
        if ~isempty(obj.pixel_mask)
            refs = obj.pixel_mask.export(fid, [fullpath '/pixel_mask'], refs);
        end
        if ~isempty(obj.pixel_mask_index)
            refs = obj.pixel_mask_index.export(fid, [fullpath '/pixel_mask_index'], refs);
        end
        refs = obj.reference_images.export(fid, [fullpath '/reference_images'], refs);
        if ~isempty(obj.voxel_mask)
            refs = obj.voxel_mask.export(fid, [fullpath '/voxel_mask'], refs);
        end
        if ~isempty(obj.voxel_mask_index)
            refs = obj.voxel_mask_index.export(fid, [fullpath '/voxel_mask_index'], refs);
        end
    end
end

end