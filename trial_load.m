%%Load all subjects files and arrange data
%makes array of all subjects obj.

 PATH = "C:\Users\User\Documents\2017-2018\Project\data\007\" ;%***Need to add: path to folder
global BLOCKS; 
BLOCKS = 26;

global num_levels;
num_levels = 6;

SUBJECTS = 5; %***Need to change- # of subjects
CHANNELS = 2; %ECG + GSR

%For the network: Data's parameters
over_lap = 0;
seg_len = 2200;
train_per = 0.8;
%     %Load ECG files: stress + noStress
%   (ST)
    file_name = strcat(PATH, "007_BAT_ST_ECG");
    ecg_st = load(file_name);
    %(NS)
    file_name = strcat(PATH, "007_BAT_NS_ECG");
    ecg_ns = load(file_name);   
    
    %%Parse all the blocks times and parameters for future labels with the ecg data
    %More details explained in make_blocks() function
    blocks = make_blocks(ecg_ns, ecg_st); 
    
    %%Load GSR files: stress + noStress
    %(ST)
    file_name = strcat(PATH, "007_BAT_ST_GSR");
    gsr_st = load(file_name);
    %(NS)
    file_name = strcat(PATH, "007_BAT_NS_GSR");
    gsr_ns = load(file_name);
    
    %Filter the data
    %**Need to add-- (Or OR Yonit)
    
    %%Down GSR's sampling rate
    %Takes GSR data
    gsr_ns_data = gsr_ns.data; %**Need to get valid files
    gsr_st_data = gsr_st.data;
    
    %Down the sampling rate: From 1000 hz to 250 hz
    gsr_ns_down = downsample(gsr_ns_data,4);
    gsr_st_down = downsample(gsr_st_data,4);
    
    %Add zeros to start for sync GSR with ECG data
    %Take starting time of the first block:
    %(NS): 
    start_t = blocks{1,1}(1,1);
    gsr_ns_down = [zeros(1,start_t-1), gsr_ns_down];
    
    %(ST): 
    start_t = blocks{2,1}(1,1);
    gsr_st_down = [zeros(1,start_t-1), gsr_st_down];
    
    %Takes GSR and ECG data and combine the two condition per each
    gsr_data{1,1} = gsr_ns_down;
    gsr_data{2,1} = gsr_st_down;
    ecg_data{1,1} = double(ecg_ns.EEG.data);
    ecg_data{2,1} = double(ecg_st.EEG.data);
    
    %Make subject obj. and add to subjects array
    subjects(1) = subject(blocks, ecg_data, gsr_data);
    
%**********************************%
%%Arrange the data to the network:
%Split the data into segments: 

counter = zeros(num_levels);
data = cell(num_levels,1);
labels = cell(num_levels,1);

%Loop for all subjects in : subjects and call 'Parser_levels'
for i = 1:2
    [tmp_data, tmp_labels, tmp_counter] = parser_levels(subjects, seg_len, over_lap);
    %Update data and labels
    for row = 1:num_levels
        last = counter(1,row); %Number of segments we have so far 
        to_add = tmp_counter(1,row); %Number of segments we want to add 
        data(row, last+1: last + to_add) = tmp_data(row,1:to_add); %Add new segments in the row
        labels(row, last+1: last + to_add) = tmp_labels(row,1:to_add); %Add new segments in the row
    end
    %Update the global counter
    counter = counter + tmp_counter; 
end

%Split data and labels into train and test
[train_input, train_labels, test_input, test_labels] = divide_rand(data, labels, counter, train_per); 
