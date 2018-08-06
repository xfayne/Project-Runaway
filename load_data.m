%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LOAD_DATA:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This script loads all subjects files and arranges the data as inputs and labels to the
%network (in Python).
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PARAMETERS:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Experiment's design:
    %# of blocks per condition: (Stress/ No Stress)
    global BLOCKS; 
    BLOCKS = 26;
    %# of difficulty levels: (include base level)
    global num_levels; 
    num_levels = 6;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Data's parameters:   
    %# of subjects:
    SUBJECTS = 2; 
    %Number of channels (signals): ECG + GSR:    
    CHANNELS = 2; 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For the network: inputes' parameters:
    %Train percentage:
    train_per = 0.8; 
    %window size: the input for network:
    seg_len = 2200; 
    %Over lap for the windows:
    over_lap = 0; 
    
    %RANDOM_ALL = true: if we want to split the data into train and test in random order
    %Which means we mix all the subjects'es data 
    %RANDOM_ALL = false: split subjects'es data into train or test separately
    RANDOM_ALL = false;
    
    %The number of subjects will use for train data if RANDOM_ALL == false
    %(train_per of the subjects number rounded)
    bound = -1;
    if RANDOM_ALL == false
        bound = round(SUBJECTS*train_per);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Files' parameters
    PATH = 'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\'; 
    %Files' names:
    GSR_FILENAME_ST = '_BAT_ST_GSR'; 
    GSR_FILENAME_NS = '_BAT_NS_GSR';
    ECG_FILENAME_ST = '_BAT_ST_ECG';
    ECG_FILENAME_NS = '_BAT_NS_ECG';

    %**Isn't used for now..**
    COUNTER_SUB = zeros(SUBJECTS,BLOCKS*2); %COUNTER_SUB(i,j) == 1 
    % <=> subject i is loaded correctly and the data in the block j is in use
    % 1<=j<=BLOCKS : block j in 'no Stress' condition
    % BLOCKS+1<=j<=BLOCKS*2 : block j-BLOCKS in 'Stress' condition

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Make array of all subjects obj.
%For each subject:
    %Load the 4 files of data
    %Pre-processing the signals
    %Create subject obj. with: Blocks array (contains times etc.) and the signals
    %Add to subjects array subject object 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Loop for all subjects
for i = 1:SUBJECTS

    %Subject's number:
    sub_num = '00%d';
    sub_num = sprintf(sub_num,i+6);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Load ECG files: stress + noStress:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(***Need to add checking if the file wasn't loaded correctly****)
    %(ST)
    file_name = [PATH, sub_num, ECG_FILENAME_ST];
    file_name = join(file_name);
    ecg_st = load(file_name);
    %(NS)
    file_name = [PATH, sub_num, ECG_FILENAME_NS];
    file_name = join(file_name);
    ecg_ns = load(file_name);   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Filter the ECG data:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Remove Trends from Data
    %(https://www.mathworks.com/help/signal/ug/remove-trends-from-data.html)
    %%Eliminate linear trend
    %(NS)
    dt_ecgl = detrend(ecg_ns.EEG.data);
    ecg_ns.EEG.data = dt_ecgl;
    %(ST)
    dt_ecg2 = detrend(ecg_st.EEG.data);
    ecg_st.EEG.data = dt_ecg2;
    %%Eliminate the nonlinear trend 
    opol = 6;
    %(NS)
    t1 = (1:length(ecg_ns.EEG.data));
    [p,s,mu] = polyfit(t1,ecg_ns.EEG.data,opol);
    f_y1 = polyval(p,t1,[],mu);
    dt_ecgnl = ecg_ns.EEG.data - f_y1;
    ecg_ns.EEG.data = dt_ecgnl;
    %(ST)
    t2 = (1:length(ecg_st.EEG.data));
    [p,s,mu] = polyfit(t2,ecg_st.EEG.data,opol);
    f_y2 = polyval(p,t2,[],mu);
    dt_ecgn2 = ecg_st.EEG.data - f_y2;
    ecg_st.EEG.data = dt_ecgn2;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Parse all the blocks' times and parameters:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(More details explained in make_blocks() function)
    blocks = make_blocks(ecg_ns, ecg_st); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Load GSR files: stress + noStress:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(ST)
    file_name = [PATH, sub_num, GSR_FILENAME_ST];
    file_name = join(file_name);
    gsr_st = load(file_name);
    %(NS)
    file_name = [PATH, sub_num, GSR_FILENAME_NS]; 
    file_name = join(file_name);
    gsr_ns = load(file_name);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Filter GSR data:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    smoothdata(gsr_ns.data,'movmean',20); %**Maybe we want to change window size or add other filters
    smoothdata(gsr_st.data,'movmean',20);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Down GSR's sampling rate:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Takes GSR data
    gsr_ns_data = gsr_ns.data; 
    gsr_st_data = gsr_st.data;
    
    %Down the sampling rate - From 1000 hz to 250 hz:
    gsr_ns_down = downsample(gsr_ns_data,4);
    gsr_st_down = downsample(gsr_st_data,4);
    
    %Add zeros to start for sync GSR with ECG data:
    %Take starting time of the first block:
    %(NS): 
    start_t = blocks{1,1}(1,1);
    gsr_ns_down = [zeros(1,start_t-1), gsr_ns_down];
    %(ST): 
    start_t = blocks{2,1}(1,1);
    gsr_st_down = [zeros(1,start_t-1), gsr_st_down];
    
    %Takes GSR and ECG data and combine the two conditions per each
    gsr_data{1,1} = gsr_ns_down;
    gsr_data{2,1} = gsr_st_down;
    ecg_data{1,1} = double(ecg_ns.EEG.data);
    ecg_data{2,1} = double(ecg_st.EEG.data);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Make subject obj. and add to subjects array:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subjects(i) = subject(blocks, ecg_data, gsr_data);
        
    
end 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Arrange the data to the network:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Split the data into segments and labels: 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Counting how many segments we have per level:
        counter = zeros(1,num_levels); 

        %Matrix of the data's segments: (inputs). Row i = Level i-1 ;1<=i<=6
        data = cell(num_levels,1); 

        %Matrix of the segments' labels. Row i = Level i-1 ;1<=i<=6
        labels = cell(num_levels,1); 

    %Loop for all subjects in : subjects: 
    %Per each subject:
        %Call the func. 'Parser_levels' 
        %Get cell array of the data split into segments and labels 
        %Add to 'data' and 'labels'

    for i = 1:SUBJECTS
        [tmp_data, tmp_labels, tmp_counter] = parser_levels(subjects(i), seg_len, over_lap);

        %Update data and labels
        for row = 1:num_levels
            last = counter(1,row); %Number of segments we have so far 
            to_add = tmp_counter(1,row); %Number of segments we want to add 
            data(row, last+1: last + to_add) = tmp_data(row,1:to_add); %Add new segments in the row
            labels(row, last+1: last + to_add) = tmp_labels(row,1:to_add); %Add new segments in the row

        end

        %Update the global counter
        counter = counter + tmp_counter; 
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Split the data and labels into test and train: 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %If we split the data in to test and train by subjects:
        %Means RANDOM_ALL == false
        %Train data and labels:
        if i == bound && RANDOM_ALL == false
            [train_input, train_labels] = make_inputs_labels(data, labels, counter); 
            counter = zeros(1,num_levels); %Reset the counter
            %**Need to reset data and labels also
        %Test data and labels:
        elseif  i == SUBJECTS && RANDOM_ALL == false
            [test_input, test_labels] = make_inputs_labels(data, labels, counter); 
        end
    end
    
%If we split the data in to test and train not by subjects:
%Means RANDOM_ALL == true
if RANDOM_ALL == true
    %Split data and labels into train and test
    [train_input, train_labels, test_input, test_labels] = divide_rand(data, labels, counter, train_per); 
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Save .mat file includes the test and train arrays:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 save('mat_to_python.mat','train_input','train_labels','test_input','test_labels');
    