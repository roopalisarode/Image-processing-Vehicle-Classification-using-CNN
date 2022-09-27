Dataset= imageDatastore('Dataset','IncludeSubfolders',true,'LabelSource','foldernames');% to access the dataset, the categories are the names of the folders
[Training_Dataset, Validation_Dataset, Testing_Dataset] = splitEachLabel(Dataset, 0.7, 0.15, 0.15);

net=googlenet;
analyzeNetwork(net)

Input_Layer_Size = net.Layers(1).InputSize(1:2);% Reszining the datasets according to the input layer of GoogLeNet
Resized_Training_Dataset= augmentedImageDatastore(Input_Layer_Size,Training_Dataset);
Resized_Validation_Dataset=augmentedImageDatastore(Input_Layer_Size,Validation_Dataset);
Resized_Testing_Dataset=augmentedImageDatastore(Input_Layer_Size,Testing_Dataset);

Feature_Learner= net.Layers(142).Name;
Output_Classifier= net.Layers(144).Name;

Number_of_Classes = numel(categories(Training_Dataset.Labels));

New_Feature_Learner= fullyConnectedLayer(Number_of_Classes,...
    'Name','Vehicle Feature Learner', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

New_Classifier_Layer= classificationLayer('Name','Vehicle Classifier');

Network_Architecture= layerGraph(net);

New_Network= replaceLayer(Network_Architecture,Feature_Learner,New_Feature_Learner);
New_Network= replaceLayer(New_Network,Output_Classifier,New_Classifier_Layer);

analyzeNetwork(New_Network)

Minibatch_Size= 4;
Validation_Frequency=floor(numel(Resized_Testing_Dataset.Files)/Minibatch_Size);
Training_Options= trainingOptions('sgdm',...
    'MiniBatchSize',Minibatch_Size,...
    'MaxEpochs',6,...
    'InitialLearnRate',3e-4,...
    'Shuffle','every-epoch',...
    'validationData',Resized_Validation_Dataset,...
    'ValidationFrequency',Validation_Frequency,...
    'Verbose',false,...
    'Plots','training-progress');


net=trainNetwork(Resized_Testing_Dataset,New_Network,Training_Options);

