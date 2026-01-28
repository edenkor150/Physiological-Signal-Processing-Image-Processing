function [Eye_Pos]=EyePosition_FUNC(Frame)
% [Eye_Pos]=EyePosition_FUNC(Frame)
% 
% This function returns the eyes position from the input frame
% 
% Inputs:
% Frame     - Desired frame for extraction eye position. Matrix of
%           normalized gray scale image
% 
% Outputs:
% Eye_Pos   - The eye position of both of the eyes. Returns as a matrix 2x2
%           where the first row is the Left eye and second row is the Right
%           eye. in each row there is the coordinates (x and y indexes of 
%           the image)

% Assining print variable. Change to 1 for plotting image
print=0;

% Filtering the desired frame with median filter of size 45x45
adjFrame=CleanSP(Frame,'Median',45,45);
% Printing image
if print
    figure('WindowState','maximized')
    nexttile
    imshow(adjFrame)
    title('Filtered Image for Detection Eye Position')
end

% Finding circles
min_rad=20;
max_rad=30;
centers=imfindcircles(adjFrame,[min_rad max_rad],'ObjectPolarity','dark' ...
    ,'EdgeThreshold',0.01,'Method','TwoStage');
% Rounding for indices
Eye_Pos=ceil(centers);

% Sorting the circle indices acoording to their gray scale value
[~,ind]=sort(sum(adjFrame(Eye_Pos(:,2),Eye_Pos(:,1)).*eye(length(Eye_Pos))),'ascend');
% Extracing only the blackest indices
Eye_Pos=Eye_Pos(ind(1:2),:);

% Sorting the rows as left in first row and right in second
[~,I]=sort(Eye_Pos,'descend');
Eye_Pos=Eye_Pos(I(:,1),:);
end
