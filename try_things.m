%Try things:
bound = round(2*0.3)
% A = data(1,1:8)
% EEG = pop_loadset( '008_BAT_ST_ECG.set', 'C:/Users/User/Documents/2017-2018/Project/data/007/');
% 
% formatSpec = 'The array is %dx%d.';
% A1 = 2;
% A2 = 3;
% 
% sub_num = '00%d';
% str = sprintf(sub_num,A2)
%   
% y = medfilt1(gsr_st.data);
% GSR = load("C:\Users\User\Documents\2017-2018\Project\data\007\007_BAT_ST_GSR");
% A = GSR.data(1:200000);
% B = smoothdata(GSR.data(1:200000),'movmean',5); 
% C = smoothdata(GSR.data(1:200000),'movmean',20);
% % [M, f, y, y2] = fftf(gsr_st.time_index, gsr_st.data, 1);
% % 
% % GSR_data = gsr_st.data;
% % ECG_data = ecg_ns.EEG.data;
% % 
% % %%%
% 
% ECG = load("C:\Users\User\Documents\2017-2018\Project\data\007\007_BAT_ST_ECG");
% D = ECG.EEG.data(1:200000);
% 
% 
% t = (1:length(D));
% 
% dt_ecgl = detrend(D);
% %%%
% opol = 6;
% [p,s,mu] = polyfit(t,D,opol);
% f_y = polyval(p,t,[],mu);
% 
% dt_ecgnl = D - f_y;

% % MM= M(1:100000);
% % plot(1:20000,GSR_data,1:20000, B)
% %  data{6,:} =[data(6,1:29) data(6,1:27)]
%  %data{1,:} = [data(1,:) data(1,:)]
% %  data2(1) = [data(1) data(1)];
% % %[m,num_of_labels] = size(counter)
% % [train_input , train_labels, test_input , test_labels] = divide_rand( data, labels, counter, 0.8);    
% % 
% %  A = cell2mat(train_input);
% %  B = A(1,:);
% %  A = data(1,:);
% % 
% %  B = data{1,1}(2,:);
% % y{1,1}()
% % y{1,2} = A
% 
% 
% % new(1,:) = subjects.channels{1,1}{1,1}(2:2+5-1)
% % new(2,:) = 0
% % new_label = zeros(1,6)
% % new_label(1,0+1) = 1
% %  figure
% %   plot(x,y1(1:cut),x,y2(1:cut),x,y3(1:cut),x,y4(1:cut))
% % % C = cell(zeroz(5,3))
% % a = zeros(1,2)
% % b = [zeros(1,4),a]
% % % C(1) = zeros(5,3)
% % %  Blocks{1} = zeros(3, 3)
% % %  Blocks{2} = ones(2,2)
% % %  Blocks{2,2} = cell(4, 3)
% % %  Blocks{2,2}(1:1) = '0'
% % %  Blocks{2,2}(2,2:3) = '1'
% % % Blocks{1,1}(1:2,1)
% % % % A = Blocks{1}(:,1)
% % filename = '007_BAT_NS_ECG';
% % path = 'C:\Users\User\Documents\2017-2018\Project\data\007';
% % file = fullfile(path,filename);
% % EEG_NS = load(file);
% % file = fullfile(path,'007_BAT_ST_ECG');
% % EEG_ST = load(file);
% % 
% % % Blocks  = make_blocks(EEG_NS, EEG_ST);
% % 
