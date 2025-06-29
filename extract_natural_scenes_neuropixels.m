addpath(genpath('D:\Users\USER\MATLAB\Allen_Brain_Neuropixels\'))
addpath(genpath('Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\'))
addpath(genpath('D:\Users\USER\MATLAB\matnwb-main\'))

datapath = 'Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\Allen_Neuropixels_postprocessed\AllenNeuropixels\';
analpath = 'Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\Allen_Neuropixels_postprocessed\AllenNeuropixelsAnalyzed\';

pathsv = 'D:\Users\USER\MATLAB\Allen_Brain_Neuropixels\ns\';

datapath2 = 'D:\Users\USER\Shin Lab\Allen_Neuropixels_nwb_variables\';

if ~exist(pathsv, 'dir')
    mkdir(pathsv)
end

sessions = readtable(strcat(datapath, 'sessions.csv'));

%%
for ises = 1:size(sessions, 1)
    tic
    sesid = num2str(sessions{ises,1});

    disp(sesid)
    fprintf('ises %d\n', ises);

    % %% determines RS and FS
    load(strcat('Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\Allen_Neuropixels_postprocessed\AllenNeuropixelsCCG\CCG_', sesid, '.mat'), 'nwbunitsV1')

    % spike train
    nwb = nwbRead(strcat(datapath, 'session_', sesid, '\session_', sesid, '.nwb'));
    unit_ids = nwb.units.id.data.load(); % array of unit ids represented within this
    unit_times_data = nwb.units.spike_times.data.load();
    unit_times_idx = nwb.units.spike_times_index.data.load();

    % unit별 area 찾기
    sesunits = readtable(strcat(datapath, 'units_', sesid, '.csv'));
    sesunits2nwbind = zeros(size(sesunits,1),1);
    for ii = 1:size(sesunits,1)
        sesunits2nwbind(ii) = find(unit_ids==sesunits.unit_id(ii));
    end

    nwbunitstructure = cell(size(unit_ids));
    nwbunitstructure(sesunits2nwbind) = sesunits.ecephys_structure_acronym;
    nwbunitstructure(cellfun(@isempty,nwbunitstructure))={'N/A'};

    list_nwbunitsHVA = cell(1, 5);
    list_HVA_names = ["VISl", "VISrl", "VISal", "VISpm", "VISam"];
    for hva_i = 1:numel(list_HVA_names)
        list_nwbunitsHVA{hva_i} = find(strcmp(nwbunitstructure, list_HVA_names(hva_i)));
    end

    % spike train 제작

    % spike time (모든 area)
    spiketimes = cell(size(unit_ids));
    last_idx = 0;
    for ii = 1:length(unit_ids)
        unit_id = unit_ids(ii);
        start_idx = last_idx + 1;
        end_idx = unit_times_idx(ii);
        spiketimes{ii} = unit_times_data(start_idx:end_idx);
        last_idx = end_idx;
    end

    % spike train
    Tres = 0.001; % 1ms
    stlen = ceil((max(unit_times_data)+1)/Tres); % add 1s buffer/padding after the last spike timing

    % V1
    spiketrain = false(stlen, length(nwbunitsV1));
    % ststartend = [floor(min(unit_times_data)/Tres)+1 floor(max(unit_times_data)/Tres)+1];
    for ii = 1:numel(nwbunitsV1)
        spiketrain(floor(spiketimes{nwbunitsV1(ii)}/Tres)+1, ii) = true;
    end

    % higher visual areas (HVA)
    list_spiketrain_HVA = cell(1, numel(list_HVA_names));
    for hva_i = 1:numel(list_HVA_names)
        spiketrain_HVA = false(stlen, length(list_nwbunitsHVA{hva_i}));
        % ststartend = [floor(min(unit_times_data)/Tres)+1 floor(max(unit_times_data)/Tres)+1];
        for ii = 1:numel(list_nwbunitsHVA{hva_i})
            spiketrain_HVA(floor(spiketimes{list_nwbunitsHVA{hva_i}(ii)}/Tres)+1, ii) = true;
        end
        list_spiketrain_HVA{hva_i} = spiketrain_HVA;
    end

    % %% stimtable
    stimtable = readtable(strcat(datapath, 'stimulus_table_', sesid, '.csv'));

    % natural_scenes trials
    % load(strcat(pathsv, 'Rns_V1RS_', sesid, '.mat'))
    % clear nstrialframe
    nstrials = strcmp(stimtable.stimulus_name, 'natural_scenes');
    nstrialstart = stimtable.start_time(nstrials);
    nstrialframe = str2double(stimtable.frame(nstrials));
    % save(strcat(pathsv, 'Rns_V1RS_', sesid, '.mat'), "nstrialframe", '-append')

    % stimulus duration is 0.25s, iti is 0.25s.
    psthtlins = (-250:500)';
    psthtrialinds = floor(nstrialstart'/Tres)+1 + psthtlins;

    % V1
    psthns = false(length(psthtlins), nnz(nstrials), numel(nwbunitsV1));
    for ii = 1:numel(nwbunitsV1)
        tempST = spiketrain(:,ii);
        psthns(:,:,ii) = tempST(psthtrialinds);
    end

    % HVA
    list_psthns_HVA = cell(1, numel(list_HVA_names));
    for hva_i = 1:numel(list_HVA_names)
        psthns_HVA = false(length(psthtlins), nnz(nstrials), numel(list_nwbunitsHVA{hva_i}));
        for ii = 1:numel(list_nwbunitsHVA{hva_i})
            tempST = list_spiketrain_HVA{hva_i}(:,ii);
            psthns_HVA(:,:,ii) = tempST(psthtrialinds);
        end
        list_psthns_HVA{hva_i} = psthns_HVA;
    end
    clear tempST psthtrialinds

    % V1
    Rns = (1/Tres)*squeeze(mean(psthns(psthtlins>0&psthtlins<=250,:,:),1));
    Nunits = size(Rns,2);
    Ntrials = size(Rns,1);    
    save(strcat(pathsv, 'Rns_V1_test_', sesid, '.mat'), ...
        'nstrialstart', 'psthns', 'Rns', 'Nunits', 'Ntrials', 'nstrialframe')

    % HVA
    struct_HVA = struct();
    for hva_i = 1:numel(list_HVA_names)
        Rns = (1/Tres)*squeeze(mean(list_psthns_HVA{hva_i}(psthtlins>0&psthtlins<=250,:,:),1));
        Nunits = size(Rns,2);
        Ntrials = size(Rns,1);

        % struct에 HVA 변수들 저장
        struct_HVA.(list_HVA_names(hva_i)).psthns = list_psthns_HVA{hva_i};
        struct_HVA.(list_HVA_names(hva_i)).Rns = Rns;
        struct_HVA.(list_HVA_names(hva_i)).Nunits = Nunits;
        struct_HVA.(list_HVA_names(hva_i)).Ntrials = Ntrials;
        struct_HVA.(list_HVA_names(hva_i)).nstrialstart = nstrialstart;
        struct_HVA.(list_HVA_names(hva_i)).nstrialframe = nstrialframe;

    end
    save(strcat(pathsv, 'Rns_HVA_test_', sesid, '.mat'), ...
        'struct_HVA', "-v7.3")

    toc

end
