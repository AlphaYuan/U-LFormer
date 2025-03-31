%% The import lines, like in Python and Java
function[] = trackmate_for_vip(tiff_path,vip_index,public_data_path,fov)

    import java.lang.Integer

    import ij.IJ

    import fiji.plugin.trackmate.TrackMate
    import fiji.plugin.trackmate.Model
    import fiji.plugin.trackmate.Settings
    import fiji.plugin.trackmate.SelectionModel
    import fiji.plugin.trackmate.Logger
    import fiji.plugin.trackmate.features.FeatureFilter
    import fiji.plugin.trackmate.detection.LogDetectorFactory
    import fiji.plugin.trackmate.detection.DogDetectorFactory
    import fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory
    import fiji.plugin.trackmate.gui.displaysettings.DisplaySettingsIO
    import fiji.plugin.trackmate.gui.displaysettings.DisplaySettings
    import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer
    import fiji.plugin.trackmate.action.ExportTracksToXML
    import fiji.plugin.trackmate.util.TMUtils.otsuThreshold

    %% The script itself.
    % Get currently selected image

    %imp = IJ.openImage('https://fiji.sc/samples/FakeTracks.tif');
    %imp = ij.ImagePlus('C:\Users\92036\PycharmProjects\data_track\output3.tiff');
    %imp = IJ.openImage('C:\Users\92036\PycharmProjects\data_track\output77.tiff');
    imp = IJ.openImage(tiff_path);

    %imp.show()

    dims = imp.getDimensions();
    imp.setDimensions( dims(3), dims(5), dims(4) );    
    %----------------------------
    % Create the model object now
    %----------------------------

    % Some of the parameters we configure below need to have
    % a reference to the model at creation. So we create an
    % empty model now.
    model = Model();    %imp=128*128*1*1*50

    % Send all messages to ImageJ log window.
    model.setLogger( Logger.IJ_LOGGER )

    %------------------------
    % Prepare settings object
    %------------------------

    settings = Settings( imp );

    % Configure detector - We use a java map
    settings.detectorFactory = DogDetectorFactory();
    map = java.util.HashMap();
    map.put('DO_SUBPIXEL_LOCALIZATION', true);
    map.put('RADIUS', 2);
    map.put('TARGET_CHANNEL', Integer.valueOf(1)); % Needs to be an integer, otherwise TrackMate complaints.
    map.put('THRESHOLD', 0);
    map.put('DO_MEDIAN_FILTERING', false);
    settings.detectorSettings = map;

    % Configure spot filters - Classical filter on quality.
    % All the spurious spots have a quality lower than 50 so we can add:

    trackmate = TrackMate(model, settings);
    trackmate.setNumThreads(2)


    trackmate.execDetection();
    trackmate.execInitialSpotFiltering();
    % trackmate.computeSpotFeatures(false);
    % trackmate.execSpotFiltering(false);
    % trackmate.execTracking();
    % trackmate.computeTrackFeatures(false);


    % Check if the process was successful
    ok = trackmate.checkInput();
    spots = trackmate.getModel().getSpots();

    allSpots = spots.iterator(false);
    % Initialize a container for quality data
    spotQuality = [];

    % Iterate through the spots and extract quality
    i=0
    while allSpots.hasNext()
        spot = allSpots.next();
        frame = spot.getFeature('FRAME');  % 获取斑点所在帧
        if i < 1000  % 因为帧从0开始，我们考虑前5帧即为0-4
            quality = spot.getFeature('QUALITY');
            spotQuality = [spotQuality; quality];
            i=i+1;
        end
        if i >= 1000
            break;
        end
    end

     matlabArray = zeros(size(spotQuality, 2), 1);

    for i = 1:numel(spotQuality)
        matlabArray(i) = spotQuality(i).doubleValue();
    end

    medianValue = median(matlabArray);

    filter1 = FeatureFilter('QUALITY', 4*medianValue, true);    %quality大于5的是关注的point
    %filter1 = FeatureFilter('QUALITY',3, true); 
    settings.addSpotFilter(filter1)

    % Configure tracker - We want to allow splits and fusions
    settings.trackerFactory  = SparseLAPTrackerFactory();
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings(); % almost good enough
    %settings.trackerSettings.put('ALLOW_TRACK_SPLITTING', false)
    
    %这个是原始的设定
    %settings.trackerSettings.put('ALLOW_TRACK_SPLITTING', true)
    %settings.trackerSettings.put('ALLOW_TRACK_MERGING', false)
    
    
    
    settings.trackerSettings.put('ALLOW_TRACK_SPLITTING', false)
    settings.trackerSettings.put('ALLOW_TRACK_MERGING', true)
    %%%%
      
      
    
    

    % Configure track analyzers - Later on we want to filter out tracks 
    % based on their displacement, so we need to state that we want 
    % track displacement to be calculated. By default, out of the GUI, 
    % not features are calculated. 

    % Let's add all analyzers we know of.
    settings.addAllAnalyzers()

    % Configure track filters - We want to get rid of the two immobile spots at 
    % the bottom right of the image. Track displacement must be above 10 pixels.
    %filter2 = FeatureFilter('TRACK_DISPLACEMENT', 10.0, true);
    %settings.addTrackFilter(filter2)


    %-------------------
    % Instantiate plugin
    %-------------------

    trackmate = TrackMate(model, settings);
    trackmate.setNumThreads(2)

    %--------
    % Process
    %--------

    ok = trackmate.checkInput();
    if ~ok
        display(trackmate.getErrorMessage())
    end

    ok = trackmate.process();
    if ~ok
        display(trackmate.getErrorMessage())
    end

    %----------------
    % Display results
    %----------------

    % Read the user default display setttings.
    %ds = DisplaySettingsIO.readUserDefault();

    % Big lines.
    %ds.setLineThickness( 1 )

    selectionModel = SelectionModel( model );
    %displayer = HyperStackDisplayer( model, selectionModel, imp, ds );
    %displayer.render()
    %displayer.refresh()

    % Echo results
    display( model.toString() )
    
    
    

    file = java.io.File('Cell5.xml')
    ExportTracksToXML.export(model, settings, file); 

    [tracks, metadata] = importTrackMateTracks( 'Cell5.xml' )
    
    index_x_y = csvread(vip_index);
    
    myCell_for_vip = cell(0, 1); % 创建一个空的 cell 数组
    
    num_tracks=numel(tracks)
    
    for i = 1:num_tracks
        tmp_tracks=cell2mat(tracks(i));
        if tmp_tracks(1,1)==0
            myCell_for_vip{end+1} =tmp_tracks(2:end,:);
        end
    end
    
    num_vip_tracks=numel(myCell_for_vip)
    num_index=length(index_x_y)
    
    
%     point_index=zeros(num_index,2)
%     for j = 1:num_index
%        point_index(j,1)=index_x_y(j,2) 
%        point_index(j,2)=index_x_y(j,3)  
%     end
    
    
 
%     for i = 1:num_vip_tracks
%         tmp_vip_tracks=cell2mat(myCell_for_vip(i));
%         point_track=[tmp_vip_tracks(1,2),tmp_vip_tracks(1,3)]
%         distance_cal=zeros(num_index,1)
%         for j = 1:num_index
%             point_cal=[index_x_y(j,2),index_x_y(j,3)]
%             distance = pdist2(point_track, point_cal, 'euclidean');
%             distance_cal(j)=distance
%         end
% 
%     end

    myCell_for_vip_dataset = cell(0, 1); % 创建一个空的 cell 数组
    index_tracks=zeros(num_vip_tracks,1);
    for i = 1:num_index
        point_cal=[index_x_y(i,2),index_x_y(i,3)];
        distance_cal=zeros(num_vip_tracks,1);
        for j = 1:num_vip_tracks
            tmp_vip_tracks=cell2mat(myCell_for_vip(j));
            point_track=[tmp_vip_tracks(1,2),tmp_vip_tracks(1,3)];
            distance = pdist2(point_track, point_cal, 'euclidean');
            distance_cal(j)=distance;
        end
        [minimumValue, index_tmp] = min(distance_cal);
        index_tracks(i)=index_tmp;
        tmp_vip_tracks=cell2mat(myCell_for_vip(index_tmp));
        x_tmp=tmp_vip_tracks(:,2)';
        y_tmp=tmp_vip_tracks(:,3)';
        id_tmp=index_x_y(i,1);
        long_tmp = [x_tmp, y_tmp];
        long_tmp(end+1) = id_tmp;
        myCell_for_vip_dataset{end+1} =long_tmp;       
    end
    
    
    out_path = sprintf('vip_tracks_fov_%d.csv', fov-1)
    out_path = strcat(public_data_path,out_path)
    
    % 创建一个文件句柄，用于数据写入
    fileID = fopen(out_path, 'w');

    % 使用 for 循环逐行写入数据
    for i = 1:numel(myCell_for_vip_dataset)
        fprintf(fileID, '%f, ', myCell_for_vip_dataset{i});
        fprintf(fileID, '\n');
    end

    % 关闭文件句柄
    fclose(fileID);
    
    
    myCell_for_all = cell(0, 1); % 创建一个空的 cell 数组
    
    for i = 1:num_tracks
        tmp_tracks=cell2mat(tracks(i));
        if tmp_tracks(1,1)==0
            tmp_tracks =tmp_tracks(2:end,:);
        end
        x_tmp=tmp_tracks(:,2)';
        y_tmp=tmp_tracks(:,3)';
        long_tmp = [x_tmp, y_tmp];
        long_tmp(end+1) = i;
        myCell_for_all{end+1} =long_tmp;    
    end
    
    out_path = sprintf('all_tracks_fov_%d.csv', fov-1)
    out_path = strcat(public_data_path,out_path)
    
    % 创建一个文件句柄，用于数据写入
    fileID = fopen(out_path, 'w');

    % 使用 for 循环逐行写入数据
    for i = 1:numel(myCell_for_all)
        fprintf(fileID, '%f, ', myCell_for_all{i});
        fprintf(fileID, '\n');
    end

    % 关闭文件句柄
    fclose(fileID);
    
    
    
    clear all;
    clc;
     
   
    
    
end

