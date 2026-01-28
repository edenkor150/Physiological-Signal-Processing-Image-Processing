function [EyeCalib] = EyeCalibration_FUNC(CalibFrame)
% [EyeCalib] = EyeCalibration_FUNC(CalibFrame)
% 
% This function returns the eyes position calibration matrix from the input
% 
% Inputs:
% CalibFrame    - Cell array 4x1 in which each cell is 1x12 cell array of
%               cropped frames
% 
% Outputs:
% EyeCalib      - The eye position calibration matrix of both of the eyes.
%               Returns as a matrix (N-1)x8x2 where N is the number of
%               calibration frames, 8 is the number of coordinates in each
%               position (2 [coordinates]*4 [positions]) and 2 is the 
%               dimantion of the eyes (1 for each eye for calibration).

% Assining print variable. Change to 1 for plotting image
print=0;
% Assigning variables and Preallocating memory
N=length(CalibFrame{1});
left=1;
right=2;
EyeCalib=nan(N-1,8,2);
for i=1:4
    for j=1:N-1
        % Extracting frames
        Frame=CalibFrame{i}{j};
        % Finding eye position through our position function
        Eye_Pos=EyePosition_FUNC(Frame);
        % Allocating the eye position to our output
        EyeCalib(j,[2*i-1 2*i],left)=Eye_Pos(left,:);
        EyeCalib(j,[2*i-1 2*i],right)=Eye_Pos(right,:);
        % Printing results
        if print
            nexttile
            imshow(Frame)
            hold on
            viscircles(Eye_Pos,28,'EdgeColor','b');
            plot(Eye_Pos(1,1),Eye_Pos(1,2),'r*')
            plot(Eye_Pos(2,1),Eye_Pos(2,2),'r*')
        end
    end
end
end
