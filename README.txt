In this folder there are all the functions that needed to
load the data and arrange it into segments for train and test arrays 
which will used as inputs into the network in python.

These arrays are: 
	train_input
	train_labels
	test_input	
	test_label

**needs to add details about the functions***

-ECG_save_to_mat: takes ecg files and save into mat files

Instructions:

1) Run the script " ".
After running the script all the arrays which are specified above
will be saved in mat file called: "mat_to_python.mat" 
(in this folder)

2) Run "import_and_network.py" file.
The file load "mat_to_python.mat" and start running the nwtwork. 
***Yonit can add more description***

- Load the data files for all subjects: 
	each subject has the following files:
	#SUBJECT_BAT_NS_ECG, #SUBJECT_BAT_ST_ECG,
	#SUBJECT_BAT_NS_GSR, #SUBJECT_BAT_ST_GSR.
- 