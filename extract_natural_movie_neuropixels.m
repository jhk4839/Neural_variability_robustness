addpath(genpath('D:\Users\USER\MATLAB\Allen_Brain_Neuropixels\'))
addpath(genpath('Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\'))
addpath(genpath('D:\Users\USER\MATLAB\matnwb-main\'))

datapath = 'Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\Allen_Neuropixels_postprocessed\AllenNeuropixels\';
analpath = 'Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\Allen_Neuropixels_postprocessed\AllenNeuropixelsAnalyzed\';

pathsv = 'D:\Users\USER\MATLAB\Allen_Brain_Neuropixels\nm\';

datapath2 = 'D:\Users\USER\Shin Lab\Allen_Neuropixels_nwb_variables\';

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
    
        % %% determines RS and FS
        load(strcat('Z:\BerkeleyGoogleDriveBackup\DATA\OpenSource\AllenBrainNeuropixels\Allen_Neuropixels_postprocessed\AllenNeuropixelsCCG\CCG_', sesid, '.mat'), 'nwbunitsV1')
    
        % spike train
        nwb = nwbRead(strcat(datapath, 'session_', sesid, '\session_', sesid, '.nwb'));
        unit_ids = nwb.units.id.data.load(); % array of unit ids represented within this
        unit_times_data = nwb.units.spike_times.data.load();
        unit_times_idx = nwb.units.spike_times_index.data.load();

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

        % stimtable
        stimtable = readtable(strcat(datapath, 'stimulus_table_', sesid, '.csv'));

        %natural_moive trials
        % load(strcat(pathsv, 'Rns_V1RS_', sesid, '.mat'))
        % clear nmtrialframe
        if strcmp(sessions{ises, 'session_type'}{:}, 'brain_observatory_1.1')
            nmtrials = strcmp(stimtable.stimulus_name, 'natural_movie_one');

            % movie one
            fps = 30; % Hz
            moviedur = 30; % sec
            n_movierepeat = 20;

            % % movie three
            % fps = 30; % Hz
            % moviedur = 120; % sec
            % n_movierepeat = 10;
        else
            % % movie three
            % continue;

            % movie one
            nmtrials = strcmp(stimtable.stimulus_name, 'natural_movie_one_more_repeats');
            fps = 30; % Hz
            moviedur = 30; % sec
            n_movierepeat = 60;
        end
        nmtrialstart = stimtable.start_time(nmtrials);
        nmtrialstop = stimtable.stop_time(nmtrials);
        nmtrialframe = str2double(stimtable.frame(nmtrials));

        % % movie one이 잘 상영되지 않은 세션 있음...stimtable.stimulus_name이 'invalid_presentation' (5, 14, 43번 세션)
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

        % stimulus duration is 0.25s, iti is 0.25s.    
        psthtlinm = (-250:500)';

        % frame 개수 기준으로 끊기
        twin = 400;
        n_tt_frame = round(twin/33); % 하나의 trial type으로 묶을 frame 개수; 반올림해도 되는지 확인!

        nmtrialstart = nmtrialstart(1:n_tt_frame:size(nmtrialstart, 1));
        nmtrialframe = nmtrialframe(1:n_tt_frame:size(nmtrialframe, 1));
        psthtrialinds = floor(nmtrialstart'/Tres)+1 + psthtlinm;
        psthnm = false(length(psthtlinm), size(nmtrialstart, 1), numel(nwbunitsV1));

        for ii = 1:numel(nwbunitsV1)
            tempST = spiketrain(:,ii);
            psthnm(:,:,ii) = tempST(psthtrialinds);
        end
        clear tempST psthtrialinds

        % trial별 firing rate 저장
        pathsv2 = strcat(pathsv, num2str(n_tt_frame), '_frame\');
        if ~exist(pathsv2, 'dir')
            mkdir(pathsv2)
        end

        Rnm = (1/Tres)*squeeze(mean(psthnm(psthtlinm>0&psthtlinm<=twin,:,:),1));
        Nunits = size(Rnm,2);
        Ntrials = size(Rnm,1);
        save(strcat(pathsv2, 'Rnm_V1_', sesid, '.mat'), ...
            'nmtrialstart', 'psthnm', 'Rnm', 'Nunits', 'Ntrials', 'nmtrialframe', '-v7.3')

        toc
    end
end