Dataset= imageDatastore('Dataset','IncludeSubfolders',true,'LabelSource','foldernames');% to access the dataset, the categories are the names of the folders
[Training_Dataset, Validation_Dataset, Testing_Dataset] = splitEachLabel(Dataset, 0.7, 0.15, 0.15);

net=googlenet;
analyzeNetwork(net)