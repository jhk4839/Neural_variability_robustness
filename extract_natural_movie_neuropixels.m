% Set appropriate path to your directory
addpath(genpath('D:\Users\USER\MATLAB\Allen_Brain_Neuropixels\'))
addpath(genpath('Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\'))
addpath(genpath('D:\Users\USER\MATLAB\matnwb-main\'))

datapath = 'Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\Allen_Neuropixels_postprocessed\AllenNeuropixels\';

% Set your save path
pathsv = 'D:\Users\USER\MATLAB\Allen_Brain_Neuropixels\nm\';

if ~exist(pathsv, 'dir')
    mkdir(pathsv)
end

sessions = readtable(strcat(datapath, 'sessions.csv'));

%%
for ises = 1:size(sessions, 1)
    if ~ismember(ises, [5 14 43]) % movie one truncated
        tic
        sesid = num2str(sessions{ises,1});
    
        disp(sesid)
        fprintf('ises %d\n', ises);
    
        % spike train
        nwb = nwbRead(strcat(datapath, 'session_', sesid, '\session_', sesid, '.nwb'));
        unit_ids = nwb.units.id.data.load(); % array of unit ids represented within this
        unit_times_data = nwb.units.spike_times.data.load();
        unit_times_idx = nwb.units.spike_times_index.data.load();

        % brain area for units
        sesunits = readtable(strcat(datapath, 'units_', sesid, '.csv'));
        sesunits2nwbind = zeros(size(sesunits,1),1);
        for ii = 1:size(sesunits,1)
            sesunits2nwbind(ii) = find(unit_ids==sesunits.unit_id(ii));
        end
    
        nwbunitstructure = cell(size(unit_ids));
        nwbunitstructure(sesunits2nwbind) = sesunits.ecephys_structure_acronym;
        nwbunitstructure(cellfun(@isempty,nwbunitstructure))={'N/A'};

        % V1
        nwbunitsV1 = find(strcmp(nwbunitstructure, 'VISp'));
        nwbunitsDVpos = zeros(size(unit_ids));
        nwbunitsDVpos(sesunits2nwbind) = sesunits.dorsal_ventral_ccf_coordinate;
        nwbunitsV1DVpos = nwbunitsDVpos(nwbunitsV1);
        [sv,si]=sort(nwbunitsV1DVpos);
        nwbunitsV1 = nwbunitsV1(si);

        % create spike train

        % spike time (all areas)
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

        % stimtable
        stimtable = readtable(strcat(datapath, 'stimulus_table_', sesid, '.csv'));

        % natural_moive trials
        for movie_name = 1:2:3
            if movie_name == 1
                % load(strcat(pathsv, 'Rns_V1RS_', sesid, '.mat'))
                % clear nmtrialframe
                if strcmp(sessions{ises, 'session_type'}{:}, 'brain_observatory_1.1')
                    nmtrials = strcmp(stimtable.stimulus_name, 'natural_movie_one');
                    
                    % movie one
                    fps = 30; % Hz
                    moviedur = 30; % sec
                    n_movierepeat = 20;
                else
                    % movie one
                    nmtrials = strcmp(stimtable.stimulus_name, 'natural_movie_one_more_repeats');
                    fps = 30; % Hz
                    moviedur = 30; % sec
                    n_movierepeat = 60;
                end
                nmtrialstart = stimtable.start_time(nmtrials);
                nmtrialstop = stimtable.stop_time(nmtrials);
                nmtrialframe = str2double(stimtable.frame(nmtrials));
        
                % % movie one presentation had trouble in some trials...stimtable.stimulus_name 'invalid_presentation' (session 5, 14, 43)
                % if size(nmtrialframe, 1) ~= fps * moviedur * n_movierepeat
                %     % invalid_pres = false(size(stimtable, 1), 1);
                %     % for i = 1:size(stimtable, 1)
                %     %     if strcmp(stimtable{i, 'stimulus_name'}, 'invalid_presentation')
                %     %         invalid_pres(i) = true;
                %     %     end
                %     % end
                % 
                %     fprintf('ises %d, number of frames %d\n', ises, size(nmtrialframe, 1));
                %     disp("Movies were truncated.")
                %     continue;
                % end
        
                % record psth from -250 to 500 ms; not important because we will cut using 'twin'
                psthtlinm = (-250:500)';
        
                % number of frames
                twin = 400;
                n_tt_frame = round(twin/33); % number of frames regarded as one stimlus
        
                nmtrialstart = nmtrialstart(1:n_tt_frame:size(nmtrialstart, 1));
                nmtrialframe = nmtrialframe(1:n_tt_frame:size(nmtrialframe, 1));
                psthtrialinds = floor(nmtrialstart'/Tres)+1 + psthtlinm;
                psthnm = false(length(psthtlinm), size(nmtrialstart, 1), numel(nwbunitsV1));
        
                for ii = 1:numel(nwbunitsV1)
                    tempST = spiketrain(:,ii);
                    psthnm(:,:,ii) = tempST(psthtrialinds);
                end
                clear tempST psthtrialinds
        
                % save path
                pathsv2 = strcat(pathsv, num2str(n_tt_frame), '_frame\');
                if ~exist(pathsv2, 'dir')
                    mkdir(pathsv2)
                end
        
                Rnm = (1/Tres)*squeeze(mean(psthnm(psthtlinm>0&psthtlinm<=twin,:,:),1));
                Nunits = size(Rnm,2);
                Ntrials = size(Rnm,1);
                save(strcat(pathsv2, 'Rnm_V1_', sesid, '.mat'), ...
                    'nmtrialstart', 'psthnm', 'Rnm', 'Nunits', 'Ntrials', 'nmtrialframe', '-v7.3')
                    
            else
                % load(strcat(pathsv, 'Rns_V1RS_', sesid, '.mat'))
                % clear nmtrialframe
                if strcmp(sessions{ises, 'session_type'}{:}, 'brain_observatory_1.1')
                    nmtrials = strcmp(stimtable.stimulus_name, 'natural_movie_three');

                    % movie three
                    fps = 30; % Hz
                    moviedur = 120; % sec
                    n_movierepeat = 10;
                else
                    % movie three
                    continue;
                end
                nmtrialstart_three = stimtable.start_time(nmtrials);
                nmtrialstop = stimtable.stop_time(nmtrials);
                nmtrialframe_three = str2double(stimtable.frame(nmtrials));
        
                % record psth from -250 to 500 ms; not important because we will cut using 'twin'
                psthtlinm = (-250:500)';
        
                % number of frames
                twin = 400;
                n_tt_frame = round(twin/33); % number of frames regarded as one stimlus
        
                nmtrialstart_three = nmtrialstart_three(1:n_tt_frame:size(nmtrialstart_three, 1));
                nmtrialframe_three = nmtrialframe_three(1:n_tt_frame:size(nmtrialframe_three, 1));
                psthtrialinds = floor(nmtrialstart_three'/Tres)+1 + psthtlinm;
                psthnm_three = false(length(psthtlinm), size(nmtrialstart_three, 1), numel(nwbunitsV1));
        
                for ii = 1:numel(nwbunitsV1)
                    tempST = spiketrain(:,ii);
                    psthnm_three(:,:,ii) = tempST(psthtrialinds);
                end
                clear tempST psthtrialinds
        
                % save path
                pathsv2 = strcat(pathsv, num2str(n_tt_frame), '_frame\');
                if ~exist(pathsv2, 'dir')
                    mkdir(pathsv2)
                end
        
                Rnm_three = (1/Tres)*squeeze(mean(psthnm_three(psthtlinm>0&psthtlinm<=twin,:,:),1));
                Nunits = size(Rnm_three,2);
                Ntrials_three = size(Rnm_three,1);
                save(strcat(pathsv2, 'Rnm_V1_', sesid, '.mat'), ...
                    'nmtrialstart_three', 'psthnm_three', 'Rnm_three', 'Nunits', 'Ntrials_three', 'nmtrialframe_three', '-append', '-v7.3')
        
            end
        end
        
        toc
    end
end