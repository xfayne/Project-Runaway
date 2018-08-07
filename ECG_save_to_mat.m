
SUBJECTS = 5;

%Array of subjects' numbers that we won't load their files
    notInclude = [2,3,4];
    %Number of valid subjects:
    valSUBJECTS = length(notInclude);

PATH = 'C:/Users/User/Documents/2017-2018/Project/data/007/';

ECG_FILENAME_ST = '_BAT_ST_ECG';
ECG_FILENAME_NS = '_BAT_NS_ECG';

 for i = 1:SUBJECTS
    if ismember(i,notInclude) == 0
        %Subject's number
        sub_num = '00%d';
        sub_num = sprintf(sub_num,i);

        %%Load ECG .set files and save to .mat files: stress + noStress 
        %(ST)
        %Create the file name
        file_name = [sub_num, ECG_FILENAME_ST];
        %The file name of .set 
        file_name_set = [file_name,'.set'];
        file_name_set = join(file_name_set);
        %Load the .set file
        EEG = pop_loadset(file_name_set, PATH);
        %The file name of .mat 
        file_name_mat = [file_name,'.mat'];  
        file_name_mat = join(file_name);
        %Save the .mat file
        save(file_name_mat,'EEG');

        %(NS)
        %Create the file name
        file_name = [sub_num, ECG_FILENAME_NS];
        %The file name of .set 
        file_name_set = [file_name,'.set'];
        file_name_set = join(file_name_set);
        %Load the .set file
        EEG = pop_loadset(file_name_set, PATH);
        %The file name of .mat 
        file_name_mat = [file_name,'.mat'];
        file_name_mat = join(file_name);
        %Save the .mat file
        save(file_name_mat,'EEG');

    end
 end