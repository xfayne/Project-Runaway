function [ data, labels, counter, StNoSt_labels ] = parser_levels( subject, seg_len, over_lap, filesLoaded, sub_num, flag_St )
%Gets subject and the length of segment window and over lap percantage in
%   fractions (0.5, 0.0 etc.), filesLoaded and the subject number. Also
%   flagSt == 1 <=> we want array of Stress/NoStress labels.
%   Returns: 1) The data: Array of matrix (In ascending order by the different levels)
%               each matrix size is: 2x(seg_len): row 1: ecg voltages, row 2: gsr voltages 
%            2) The labels: Array of 6 length vectors (**Maybe we want to change to other structure) 
%               level 1 = (010000) etc.
%            3) Counter array: in length of 6: counting the number of
%            segments we have per each level
%            4) StNoSt_labels optional. return array of Stress/NoStress
%            labels: (1,0) == NoStress and (0,1) == Stress
        
        num_levels = 6; %**Need to check global
        BLOCKS = 26;
        
        
        data = cell(num_levels,1);
        labels = cell(num_levels,1);
        StNoSt_labels = cell(num_levels,1);
        counter = zeros(1,num_levels);
        
        for cond = 1:2 %Stress and NoStress
            for curr_b = 1:BLOCKS %Number of the blocks
                if filesLoaded(cond, sub_num) == 1 %Condition's files are loaded
                    level = subject.blocks{cond,1}(curr_b,3);                   
                    start_t = subject.blocks{cond,1}(curr_b,1);
                    end_t = subject.blocks{cond,1}(curr_b,2);
                    back = seg_len*over_lap;
                    ind = start_t;
                    %Make segments from the current block
                    while ind + seg_len - 1 <= end_t 
                        %ECG
                        new_seg(1,:) =  subject.channels{1,1}{cond,1}(ind:ind+seg_len-1);
                        %GSR
                        new_seg(2,:) =  subject.channels{2,1}{cond,1}(ind:ind+seg_len-1);
                        %Label : vector represents the level
                        new_label = zeros(1,num_levels);
                        new_label(1,level+1) = 1; 
                        
                        %Label : tuple represents the stress/NoStress cond                      
                        if flag_St == 1
                            new_st_label = zeros(1,2);
                            new_st_label(1,cond) = 1; 
                        end
                        %Points after the last index in the level
                        last = counter(1,level+1) + 1; 
                        %Append to data and labels
                        data{level+1,last}(:,:) = new_seg;
                        labels{level+1,last}(:,:) = new_label;
                        %Append to tress/NoStress labels
                        if flag_St == 1
                            StNoSt_labels{level+1,last}(:,:) = new_st_label;
                        end
                        %Update the counter
                        counter(1,level+1) = last;
                        %Jump to next segment's starting index
                        ind = ind + seg_len - back;
                    end
                end
            end
        end                  
end
