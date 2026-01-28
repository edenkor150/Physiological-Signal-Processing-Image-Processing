% Adjusting the default figure settings
set(0,'defaultAxesXGrid','on') 
set(0,'defaultAxesYGrid','on')
print=0;
%% EXP 1
% 1.1
% Loading blood platelets picture
rice=imread('rice.png');
rice=mat2gray(rice);
% Showing the picture
figure
imshow(rice)
title('Blood Platelets Photo')

% 1.2
% Plotting histogram
figure
imhist(rice)
title('Histogram of Gray Levels - Original Photo')
xlabel({'','','','Gray Levels'})
ylabel('Count')
lim=ylim;
ylim(lim+[0 500])

% 1.3
% Creating mask manualy
TH=0.5;
riceMask=rice;
riceMask(riceMask>=TH)=1;
riceMask(riceMask<TH)=0;
% Showing the mask
figure
imshow(riceMask)
title('Rice Mask - Manual Threshold')

% 1.4
% Cheacking menualy what are the dimentions of the blood platlets
h=imdistline;
delete(h)
% Creating the disk structure with the smaller blood platlets radius
disk=strel('disk',10);
% Using erosion to remove the blood platelets
notRice=imerode(rice,disk);
% Showing the background
figure
imshow(notRice)
title('Background')
% Removing background
onlyRice=rice-notRice;
figure
imshow(onlyRice)
title('Only the Blood Platlets')

% 1.5
% Plotting new histogram
figure
imhist(onlyRice)
title('Histogram of Gray Levels - After Filtering')
xlabel({'','','','Gray Levels'})
ylabel('Count')
lim=ylim;
ylim(lim+[0 1500])
% New threshold
TH=0.35;
riceMaskOR=onlyRice;
riceMaskOR(riceMaskOR>=TH)=1;
riceMaskOR(riceMaskOR<TH)=0;
% Plotting new mask
figure
imshow(riceMaskOR)
title('Blood Platelets Mask - After Background Removal')

% Edge detection dilation-mask
edgeD_M=imdilate(riceMaskOR,strel('diamond',1))-riceMaskOR;
% Plotting the results
figure
nexttile
imshow(edgeD_M)
title('Blood Platelets After Dilation-Mask')
% Edge detection laplacian of gaussian
nexttile
edge(onlyRice,'log')
title('Blood Platelets After LoG')

% 1.6
% Calculating number of blood plateles
[ricEdges,nObjs]=bwlabel(imclearborder(edgeD_M));
figure
imshow(ricEdges)
title(['Clean Edge Detection - Number of Blood Plateles=' num2str(nObjs)])

% 1.7
% Finding the blood platlets with right oriantation
orientInfo=regionprops(ricEdges,'Orientation','FilledImage','BoundingBox');
RightOrientInfo=orientInfo([orientInfo.Orientation]>0);
% Creating The mask. Starting with blank photo
RightMask=zeros(size(rice));
for i=1:length(RightOrientInfo)
    % Indexes for up left corner
    LeftCornerX=ceil(RightOrientInfo(i).BoundingBox(1));
    LeftCornerY=ceil(RightOrientInfo(i).BoundingBox(2));
    % All the indices for the right orianted blood platletes
    indX=LeftCornerX:LeftCornerX+RightOrientInfo(i).BoundingBox(3)-1;
    indY=LeftCornerY:LeftCornerY+RightOrientInfo(i).BoundingBox(4)-1;
    % Adding back the parts containing the full right blood platlets
    RightMask(indY,indX)=RightMask(indY,indX)+RightOrientInfo(i).FilledImage;
end
% Showing the mask
figure
imshow(RightMask)
title('Only Right Orianted Blood Plateles - Mask')

% 1.8
% Applying the mask to the original photo
OnlyRightRice=rice.*RightMask;
% Showing results
figure
imshow(OnlyRightRice)
title('Only Right Orianted Blood Plateles - After Masking')

%% Exp 2
% Loading photos
RegVeg=imread('images\vegetables.tif');
RegVeg=mat2gray(RegVeg);
NoiseVeg=imread('images\Nvegetables.tif');
NoiseVeg=mat2gray(NoiseVeg);
% Showing the photos
figure
nexttile
imshow(RegVeg)
title('Vegtables')
nexttile
imshow(NoiseVeg)
title('Noisy Vegtables')

% 2.1
% Filter dimentions
var1=[1 1 2 3 3 4 4 5 7];
var2=[2 3 2 3 4 4 5 5 7];
% Allocating memory
MSE=zeros(length(var2),1);
figure(WindowState="maximized")
for i=1:length(var2)
    % Median filtering
    FilteredVeg=CleanSP(NoiseVeg,'Median',var1(i),var2(i));
    % Calculating MSE
    MSE(i)=mean((FilteredVeg-RegVeg).^2,"all");
    % Showing results
    nexttile
    imshow(FilteredVeg)
    title({['Median Filter ' num2str(var1(i)) 'x' num2str(var2(i))],['MSE=' num2str(MSE(i))]},FontSize=16)
end
% Plotting MSE graph
figure
plot(MSE)
title('MSE for Various Median Filters Sizes')
xlabel('Median Size')
ylabel('MSE')
xticklabels({'1x2' '1x3' '2x2' '3x3' '3x4' '4x4' '4x5' '5x5' '7x7'})

% 2.2
% Allocating memory
MSE=zeros(100,length(var2));
% Loop for 100 different noise and filtering simulation
for i=1:100
    % Adding salt&peper noise - 20%
    SPNoisyVeg=imnoise(RegVeg,'salt & pepper',0.2);
    for j=1:length(var2)
        % Filtering with all the median filters sizes
        FilteredVeg=CleanSP(SPNoisyVeg,'Median',var1(j),var2(j));
        % Calculating MSE
        MSE(i,j)=mean((FilteredVeg-RegVeg).^2,"all");
    end
end
% Calculating maen and std
meanMSE=mean(MSE);
stdMSE=std(MSE);

% Plotting boxplot
figure(WindowState="maximized")
boxplot(MSE,'Labels',{'1x2' '1x3' '2x2' '3x3' '3x4' '4x4' '4x5' '5x5' '7x7'})
title('MSE boxplot',FontSize=16)
xlabel('Median Filter Size',FontSize=14)
ylabel('MSE',FontSize=14)

% Statistical test for squere filteres
squereMSE=MSE(:,var1==var2);
% One-way ANOVA statistical test
[p,tbl]=anova1(squereMSE,[],'off');
disp(tbl)
disp(['p-Value=' num2str(p)])

%% Exp 3
% 3.1
% Load Data
ImageFolder='image processing lab\eyes\';
% Get a list of all files in the folder with the desired file name pattern
ImagData=dir(fullfile(ImageFolder,'*.jpg')); 

lenIm=length(ImagData);
% Preallocate a cell array to hold the images
EyeImag=cell(lenIm, 1);
for i=1:lenIm
    % Read the current image
    imgPath=fullfile(ImageFolder,ImagData(i).name);
    img=imread(imgPath);
    img=im2gray(img);
    img=mat2gray(img);

    EyeImag{i}=img;
end

% 3.3
% Preallocating
N=lenIm/4;
CalibFrames={cell(1,N);cell(1,N);cell(1,N);cell(1,N)};
% 3.3.1
% Creating CalibFrame cell array
for i=1:lenIm
    Frame=EyeImag{i};
    Frame=imcrop(Frame,[1103.5 343.5 702 129]);
    
    CalibFrames{mod(i-1,4)+1}{ceil(i/4)}=Frame;
end

% 3.1
% Plotting an example
Eye_Pos=EyePosition_FUNC(Frame);
% Plotting
figure
imshow(Frame)
hold on
viscircles(Eye_Pos,28,'EdgeColor','b');
plot(Eye_Pos(1,1),Eye_Pos(1,2),'r*')
plot(Eye_Pos(2,1),Eye_Pos(2,2),'r*')
title('Eye Detection Example')

% 3.3
% Calculating the calibration matrix
CalibMat=EyeCalibration_FUNC(CalibFrames);

% 3.3.2
% 3-D Histogram of Left Eye
figure
hist3(CalibMat(:,[1 2],1),'FaceColor','magenta')
hold on
hist3(CalibMat(:,[3 4],1),'FaceColor','blue')
hist3(CalibMat(:,[5 6],1),'FaceColor','yellow')
hist3(CalibMat(:,[7 8],1),'FaceColor','white')

title('Calibration 3-D Histogram - Left Eye Position')
xlabel('X index')
ylabel('Y index')
zlabel('Count')
xlim padded
ylim padded
zlim tight
legend('1','2','3','4')

% 3-D Histogram of Right Eye
figure
hist3(CalibMat(:,[1 2],2),'FaceColor','magenta')
hold on
hist3(CalibMat(:,[3 4],2),'FaceColor','blue')
hist3(CalibMat(:,[5 6],2),'FaceColor','yellow')
hist3(CalibMat(:,[7 8],2),'FaceColor','white')

title('Calibration 3-D Histogram - Right Eye Position')
xlabel('X index')
ylabel('Y index')
zlabel('Count')
xlim padded
ylim padded
zlim tight
legend('1','2','3','4')

% 3.4.2
% Testing the calibration matrix
for i=1:4
    Eye_Look=EyeLook_FUNC(CalibFrames{i}{end},CalibMat);
    disp(['Left Eye Look Number: ' num2str(Eye_Look(1))])
    disp(['Right Eye Look Number: ' num2str(Eye_Look(2))])
    disp(' ')
end

% 3.5
% Loading eye movement video
vid=VideoReader([ImageFolder 'WIN_20240123_05_11_14_Pro.mp4']);
% Segmentation
segmentation=ceil(([0 2.09,4.26,6.23,8.27,10.25,12.30,13.98,15.95,17.91 ...
    ,20.06,22.01,24.10,26.30,28.04,29.31]+1.61)*vid.FrameRate);
% Calculate look duration
lookDuration=diff(segmentation);
% The number on screen look order
numOrder=[1,2,3,2,3,4,3,2,1,2,1,2,3,1,2];
numOrderFrame=repelem(numOrder,lookDuration);

% 3.6
% For removing the beggining part of now eye movement
displace=segmentation(1);
% Preallocating memory
Eye_Look=zeros(vid.NumFrames-displace,2);
for i=displace+1:vid.NumFrames
    % Extracting frame by frame
    Frame=read(vid,i);
    Frame=im2gray(Frame);
    Frame=mat2gray(Frame);
    
    % Finding the relevant ROI
    cropPosition=[1103.5 343.5 702 129]-[320 75 702/4 129/4];
    currFrame=imcrop(Frame,cropPosition);
    % Resizing the current frame
    targetSize=size(CalibFrames{1}{1});
    currFrame=imresize(currFrame,targetSize);

    % Applying our functions
    Eye_Look(i-displace,:)=EyeLook_FUNC(currFrame,CalibMat);
end

% Plotting confusion matrixes
figure
nexttile
confusionchart(numOrderFrame,Eye_Look(:,1))
errLeft=nnz(logical(numOrderFrame.'-Eye_Look(:,1)))/length(numOrderFrame)*100;
title(['Confusion Matrix for Left Eye - Error Percentage=' num2str(errLeft)])

nexttile
confusionchart(numOrderFrame,Eye_Look(:,2))
errRight=nnz(logical(numOrderFrame.'-Eye_Look(:,2)))/length(numOrderFrame)*100;
title(['Confusion Matrix for Right Eye - Error Percentage=' num2str(errRight)])

% Plotting comparison
figure(WindowState="maximized")
plot(numOrderFrame,LineWidth=2)
hold on
scatter(1:length(numOrderFrame),Eye_Look(:,2),Marker="x")
title('Algorithm Recongnition Comparison',FontSize=16)
xlabel('Frame',FontSize=14)
ylabel('Look State',FontSize=14)
legend('True Labels','Predicted Labels Right',FontSize=13)
xlim tight
ylim padded
yticks([1 2 3 4])

%% EXP 4
% Loading the walking video
vid=VideoReader('image processing lab\WIN_20240123_14_45_03_Pro.mp4');
% 4.1
[AllLegPositionX,AllLegPositionY]=deal(zeros(4,vid.NumFrames));
for i=1:vid.NumFrames
    clc
    % Read frame
    Frame=read(vid,i);
    Frame=im2gray(Frame);
    Frame=mat2gray(Frame);
    % Creating a mask and cropping the ROI
    currFrame=imbinarize(Frame,0.4);
    currFrame=imcrop(currFrame,[589.5 380 487 440]);
    % Applying median filter
    currFrame=CleanSP(currFrame,'Median',3,3);
    % Applying imfindcircles function
    min_rad=4;
    max_rad=15;
    [centers,radii,metric]=imfindcircles(currFrame,[min_rad max_rad] ...
        ,'ObjectPolarity','bright','EdgeThreshold',0.7,'Sensitivity',0.88 ...
        ,'Method','TwoStage');
    % Find indices
    Leg_Pos=ceil(centers);

    % Sorting the centers according to Y index
    centers4=centers(1:4,:);
    [~,Ind]=sort(centers4(1:4,2));
    % Allocating centers
    AllLegPositionX(:,i)=centers(Ind,1);
    AllLegPositionY(:,i)=centers(Ind,2);
end
% 4.1.2
% Plotting walk representation
FrameEx=imcrop(Frame,[589.5 380 487 440]);
figure(WindowState="maximized")
nexttile
imshow(FrameEx)
% 4.3 Measuring marker size manualy
h=imdistline;
delete(h)
% End
hold on
scatter(AllLegPositionX(1,:),AllLegPositionY(1,:))
scatter(AllLegPositionX(2,:),AllLegPositionY(2,:))
scatter(AllLegPositionX(3,:),AllLegPositionY(3,:))
scatter(AllLegPositionX(4,:),AllLegPositionY(4,:))

title('Marker Location In Space',FontSize=16)
legend('Top Marker','2^{nd} Marker','3^{rd} Marker','Bottom Marker' ...
    ,Location='bestoutside',FontSize=13)

% 4.2
% Find the absolute velocity 
FrameTime=vid.Duration/vid.NumFrames; % Time for each frame
% Speed Calculation 
N=size(AllLegPositionX,2)-1;
% Initialize matrices to store distances and velocities
[distances,velocities]=deal(zeros(4,N));

% Calculate distances and velocities
for i=1:4 % For each location
    for j=1:N % For each frame, excluding the last one
        % Calculate distance between consecutive frames
        distances(i,j)=sqrt((AllLegPositionX(i,j+1)-AllLegPositionX(i,j))^2 ...
            +(AllLegPositionY(i,j+1)-AllLegPositionY(i,j))^2);
        velocities(i,j)=distances(i,j)/FrameTime;
    end
end
% distances matrix now contains the distance between each consecutive frame for each location
% velocities matrix contains the velocity in pixels/frame for each location

% Absolute velocities
AbsVelocities=abs(velocities);
% Mean Velociity Calculation
MeanVel=mean(AbsVelocities,2);

% Assuming velocities is a 4xN matrix with the velocities for each location
% Assuming MeanVel is a 4x1 vector with the mean velocities for each location
% Assuming FrameTime is the time for each frame

% Calculate the time points for the x-axis
time=linspace(0,vid.Duration,N);
% Create subplots for each location
figure(WindowState="maximized")
for i=1:4
    subplot(2,2,i)
    plot(time,velocities(i,:),'b') % Plot absolute velocity for location i
    hold on
    plot(time,MeanVel(i)*ones(size(time)),'r--','LineWidth',2) % Plot mean velocity line for location i
    text(2,8500,['Mean Velocity=' num2str(round(MeanVel(i),2)) ' [pixels/sec]'],color='r',FontSize=13)
    
    % Add title and labels
    title(sprintf('Marker Location %d',i),FontSize=16)
    xlabel('Time [sec]',FontSize=14)
    ylabel('Velocity [pixels/sec]',FontSize=14)
    xlim tight
    ylim([0 9000])
    % Add legend
    legend('Velocity','Mean Velocity')
end

% 4.3
% Converting from pixels to cm
MarkerSize=1; % [cm]
PixelMarkerSize=24; % [Pixels]
% Ratio
ratioPixel2Cm=MarkerSize/PixelMarkerSize;
% Calculating cumulative distace according to the bottom marker in cm
walkDist=sum(distances(4,:))*ratioPixel2Cm;
realWalkDist=2*vid.Duration*1/3600*10^5;
disp(['Walking Distance = ' num2str(walkDist) ' [cm]'])
disp(['Real Walking Distance = ' num2str(realWalkDist) ' [cm]'])

% 4.4
% Calculating distance between the top markers
hipAngleCalcX=diff([AllLegPositionX(1,:);AllLegPositionX(2,:)]).^2;
hipAngleCalcY=diff([AllLegPositionY(1,:);AllLegPositionY(2,:)]).^2;

hypoten=sqrt(hipAngleCalcX+hipAngleCalcY);
% Calculating the parralel to the hypotenuse
para=sqrt(hipAngleCalcX);

% Using a right triangle trigonometric ratio, calculate the apex angle
hipAngle=rad2deg(asin(para./hypoten));

% Calculate the time vector and plot the hip angle
time=linspace(0,vid.Duration,N+1);
figure
plot(time,hipAngle)
title('Hip Angle as a Function of Time')
xlabel('Time [sec]')
ylabel('Angle [deg]')

% Example of hip angle
Nframe=400;
Frame=read(vid,Nframe);
Frame=im2gray(Frame);
Frame=mat2gray(Frame);
Frame=imcrop(Frame,[589.5 380 487 440]);
% Example of angle detection
figure
imshow(Frame)
hold on
plot([AllLegPositionX(1,Nframe),AllLegPositionX(2,Nframe)], ...
    [AllLegPositionY(1,Nframe),AllLegPositionY(2,Nframe)],'k',LineWidth=3.5)
plot([AllLegPositionX(1,Nframe),AllLegPositionX(1,Nframe)], ...
    [AllLegPositionY(1,Nframe),AllLegPositionY(2,Nframe)],'r',LineWidth=3.5)
plot([AllLegPositionX(1,Nframe),AllLegPositionX(2,Nframe)], ...
    [AllLegPositionY(2,Nframe),AllLegPositionY(2,Nframe)],'b',LineWidth=3.5)
title('Example of Angle Detection')

% 4.5
% Calculating all the triangle legs distance
kneeAngleCalcX=diff([AllLegPositionX(1,:);
    (AllLegPositionX(2,:)+AllLegPositionX(3,:))/2; % Middle leg calculated as the mean between the 2 middle markers
    AllLegPositionX(4,:); ...
    AllLegPositionX(1,:)]).^2;
kneeAngleCalcY=diff([AllLegPositionY(1,:);
    (AllLegPositionY(2,:)+AllLegPositionY(3,:))/2; % Middle leg calculated as the mean between the 2 middle markers
    AllLegPositionY(4,:); ...
    AllLegPositionY(1,:)]).^2;

kneeAngleCalc=sqrt(kneeAngleCalcX+kneeAngleCalcY);
% Calculating the knee angle using the law of cosines
a=kneeAngleCalc(1,:);
b=kneeAngleCalc(2,:);
c=kneeAngleCalc(3,:);

kneeAngle=rad2deg(acos((a.^2+b.^2-c.^2)./(2*a.*b)));
% Plotting knee angle
figure
plot(time,kneeAngle)
title('Knee Angle as a Function of Time')
xlabel('Time [sec]')
ylabel('Angle [deg]')

% Example of knee angle
figure
Nframe=362;
Frame=read(vid,Nframe);
Frame=im2gray(Frame);
Frame=mat2gray(Frame);
Frame=imcrop(Frame,[589.5 380 487 440]);
% Example of angle detection
imshow(Frame)
hold on
plot([AllLegPositionX(4,Nframe),mean([AllLegPositionX(2,Nframe),AllLegPositionX(3,Nframe)])], ...
    [AllLegPositionY(4,Nframe),mean([AllLegPositionY(2,Nframe),AllLegPositionY(3,Nframe)])],'k',LineWidth=2)
plot([mean([AllLegPositionX(2,Nframe),AllLegPositionX(3,Nframe)]),AllLegPositionX(1,Nframe)], ...
    [mean([AllLegPositionY(2,Nframe),AllLegPositionY(3,Nframe)]),AllLegPositionY(1,Nframe)],'r',LineWidth=2)
plot([AllLegPositionX(1,Nframe),AllLegPositionX(4,Nframe)], ...
    [AllLegPositionY(1,Nframe),AllLegPositionY(4,Nframe)],'b',LineWidth=2)
title('Example of Angle Detection')
