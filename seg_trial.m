%%


%%Define: # of blocks
BLOCKS = 26;


%%Creates subject. Gets ECG and GSR files (ST and NS) and returns subject object
function subject =  creat_subject(EEG_NS, EEG_ST, acq_NS, acq_ST)

%%All bat blocks: 
%column 1: start time; 
%column 2: end time; 
%column 3: level;
Blocks = zeros(BLOCKS, 3); 

for j = 1:2

index = 1; 
foundStart = false; %Flag: if we found runStart we need to find the next runEnd 

%Loop for making a table of the blocks and parse all block's info (nLevel, ringSize etc.)
for t = EEG.event(1,:)
    if strcmp(t.type,'runStart') %Start of Block        
        B_array(index) =  parser_runStart(t.code);  %B_aray is array of all blocks objects
        Blocks(index, 1) = t.latency; %Start time of the block
        Blocks(index, 3) = level(B_array(index).nLevel, B_array(index).ringSize); %Calculate the block's level 
        foundStart = true; %Now we need to find runEnd of that block
    elseif strcmp(t.type, 'runEnd') && (foundStart) %Found start of a block and this is the block's end time
        Blocks(index, 2) = t.latency;  %Block's end time
        index = index + 1; %Search for next block
    end
   
end

B_array = transpose(B_array);
 
%%Make table of all blocks array of ecg voltages. row i = block i.
for i = 1:BLOCKS
    start = Blocks(i,1);
    endTime = Blocks(i,2);
    ind = endTime - start ;
    ecg_voltages(i,1:ind) = parser_voltage(EEG.data, start, endTime);
    %gsr_voltages(i,1:ind) = parser_voltage(acq.data(:,3), start, endTime);
    %**Needs to make gsr_voltages
end

end

%%Create a class object of subject 
subject = subject(B_array, ecg_voltages, gsr_voltages);

end
