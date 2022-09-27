[Predicted_Label, Probability]=classify(net,Resized_Testing_Dataset);
accuracy= mean(Predicted_Label==Testing_Dataset.Labels);

index= randperm(numel(Resized_Testing_Dataset.Files));
figure

for i=1:4
    subplot(2,2,i)
    I= readimage(Testing_Dataset, index(i));
    imshow(I)
    label= Predicted_Label(index(i));
    title(string(label) + ", " + num2str(100*max(Probability(index(i), :)), 3) + "%");
end
