function [Eye_Look] = EyeLook_FUNC(Frame,EyeCalib)
% [Eye_Look] = EyeLook_FUNC(Frame,EyeCalib)
% 
% This function returns the eyes position number from a single frame and a 
% specific calibration matrix
% 
% Inputs:
% Frame         - Desired frame for extraction eye position. Matrix of
%               normalized gray scale image
% EyeCalib      - The eye position calibration matrix of both of the eyes.
%               Matrix of (N-1)x8x2 where N is the number of calibration 
%               frames, 8 is the number of coordinates in each position
%               (2 [coordinates]*4 [positions]) and 2 is the dimantion
%               of the eyes (1 for each eye for calibration).
% 
% Outputs:
% Eye_Look      - A matrix of 1x2 housing the number the left and right eye
%               has looked on

% Assining print variable. Change to 1 for plotting image
print=1;
% Left and right indices
Left=1;
Right=2;
% Eye position calculation
Eye_Pos=EyePosition_FUNC(Frame);

% Calculating the mean of each index from the calibration matrix
LeftMean=mean(EyeCalib(:,:,Left));
RightMean=mean(EyeCalib(:,:,Right));
% Preallocation memory and loop for calculating the distance between the
% current eye position from the mean position for each state (number)
[LeftError,RightError]=deal(zeros(1,4));
for i=1:4
    LeftError(i)=vecnorm(Eye_Pos(Left,:)-LeftMean([2*i-1 2*i]));
    RightError(i)=vecnorm(Eye_Pos(Right,:)-RightMean([2*i-1 2*i]));
end
% Finding the minimum distance index
[~,LeftInd]=min(LeftError);
[~,RightInd]=min(RightError);
% Returning the number each eye was looking on
Eye_Look=[LeftInd,RightInd];
end
