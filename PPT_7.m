clear all
close all
warning off
clc

vid_path='Test Video path';
default_redings='csv file with ECG and PPG readings';

%///////// reading bioradio ////////////////////////////////////////////

vid_start_time=[19 38 36];
disp('reading bio signals from csv ....')
T = readtable(default_redings);
ab=table2cell(T(4:end,1));
ac=datetime(ab,'InputFormat','yyyy-MM-dd HH:mm:ss.SSS');
HH=ac.Hour;
MM=ac.Minute;
SS=ac.Second;
Time_mat=[HH MM SS];
[ri,ci]=find(Time_mat(:,1)==vid_start_time(1) & Time_mat(:,2)==vid_start_time(2) & round(Time_mat(:,3))==vid_start_time(3));
Time_mat=Time_mat(ri(1):end,:);

S=Time_mat(1:end,3);
M=Time_mat(1:end,2);
total_sec=(M*60) + S;
total_sec=total_sec-total_sec(1);

% %//////////////////////////////////////////////////////////

start_pt=0.004;
end_pt=0.05;

ecg_raw=str2double(table2cell(T(4+ri(1):end,2)));
ecg_raw=ecg_raw(round(length(ecg_raw)*start_pt):round(length(ecg_raw)*end_pt));
ecg_raw=ecg_raw - mean(ecg_raw(:));
ecg_flt=movmean(ecg_raw,10);

[pks_ecg,locs_ecg] = findpeaks(ecg_flt,'MinPeakDistance',1000);
ecg_flt_2=ecg_flt;
for i=1:length(locs_ecg)-1
    clipped_signal=ecg_flt(locs_ecg(i)+1:locs_ecg(i+1)-1);
    clipped_signal=movmean(clipped_signal,100);
    ecg_flt_2(locs_ecg(i)+1:locs_ecg(i+1)-1)=clipped_signal;
end

ppe_raw=str2double(table2cell(T(4+ri(1):end,4)));
ppe_raw=ppe_raw(round(length(ppe_raw)*start_pt):round(length(ppe_raw)*end_pt));
ppe_raw=ppe_raw - mean(ppe_raw(:));
ppe_flt=movmean(ppe_raw,20);
[pks_ppe_max,locs_ppe_max] = findpeaks(ppe_flt,'MinPeakDistance',1000);

for i=1:length(locs_ppe_max)-1
    slot_ppe=ppe_flt(locs_ppe_max(i):locs_ppe_max(i+1));
    [r,c]=find(slot_ppe==min(slot_ppe(:)));
    pks_ppe_min(i)=min(slot_ppe(:));
    locs_ppe_min(i)=locs_ppe_max(i) + r(1);
end

for i=1:length(locs_ppe_min)
    [r,c]=find(locs_ppe_max > locs_ppe_min(i));
    
    mid_val=round((locs_ppe_min(i) + locs_ppe_max(r(1)))/2);
    locs_ppe_mid(i,1)=mid_val;
    pks_ppe_mid(i,1)=ppe_flt(mid_val);
end

%///////////////////////////////////////////////////////////////////

figure(1)
subplot(3,1,1)
plot(ecg_raw)
title('ECG raw')
subplot(3,1,2)
plot(ecg_flt)
title('ECG (Moving mean)')
subplot(3,1,3)
plot(ecg_flt_2)
title('ECG (2nd filtering)')
drawnow

figure(2)
subplot(2,1,1)
plot(ppe_raw)
title('PPG raw')
subplot(2,1,2)
plot(ppe_flt)
title('PPG (Moving mean)')
drawnow

figure(3)
plot(1:length(ecg_flt_2),ecg_flt_2,'-b',1:length(ppe_flt),ppe_flt,'-c');
hold on
plot(locs_ecg,pks_ecg,'om')
hold on
plot(locs_ppe_max,pks_ppe_max,'or')
hold on
plot(locs_ppe_min,pks_ppe_min,'ok')
hold on
legend('Bio ECG','Bio PPE','Max Peaks ECG','Max Peaks PPG','Min Peaks PPG')
hold on

for i=1:length(locs_ecg)-1
    [r,c]=find(locs_ppe_max > locs_ecg(i));
    pt1=locs_ecg(i);
    pt2=locs_ppe_max(r(1));
    ptt_ref(i)=total_sec(pt2) - total_sec(pt1);
    Ptt_points(i,:)=[pt2 pt1];
    
    plot([pt1 pt1],[-1 0],'-g');
    hold on
    plot([pt2 pt2],[-1 0],'-g');
    hold on
    text((pt1+pt2)/2,-1.2,['Ptt','  ',num2str(ptt_ref(i))],'Rotation',90);
    hold on
end
hold off
drawnow
title('Calculations for PTT(ref) using ECG(ref) & PPG(ref)')

%//////// reading frames of the video and targeting areas //////////////


disp('extracting features .....')
display_flag=1;

R=15;
v = VideoReader(vid_path);
frame_rate=v.FrameRate;
per_frame_sec=1/frame_rate;
time_yet=0;
mark_flag=0;

start_index=round(length(total_sec) * start_pt);
end_index=round(length(total_sec) * end_pt);

start_time=total_sec(start_index);
end_time=total_sec(end_index);

v1 = VideoWriter('newfile.avi','Motion JPEG AVI');
v1.Quality = 100;
open(v1)

clear index_x G_mean_forehead G_mean_lefthand G_mean_ritehand
last_time=1;
first_flag=0;
frame_no=1;
while hasFrame(v)
    video_frame = readFrame(v);
   % video_frame=imrotate(video_frame,180);
    pos_hand=hand_detection(video_frame);
    
    if(first_flag==0)
        figure(4)
        imshow(video_frame);
        drawnow
        first_flag=1;
    end
    
    time_yet=time_yet + per_frame_sec;
    [rr_time,cc_time]=find(total_sec>time_yet);
    if(time_yet>end_time)
        break
    end
    
    if(time_yet>start_time && time_yet<end_time)
        
                        faceDetector = vision.CascadeObjectDetector('FrontalFaceLBP','MinSize',[200 200]);
                        box_face = step(faceDetector, video_frame);
    
                        %////////////////////////////////////////////////////////////////////

                        center_r=round(box_face(1,2) + 50);
                        center_c=round(box_face(1,1) + (box_face(1,3)/2));
                        th=0:0.1:360;
                        X_forehead=round(center_r + (R*sind(th)));
                        Y_forehead=round(center_c + (R*cosd(th)));
                        bw_forehead(1:size(video_frame,1),1:size(video_frame,2))=0;
                        bw_forehead=logical(bw_forehead);
                        for i=1:length(X_forehead)
                            bw_forehead(X_forehead(i),Y_forehead(i))=1;
                        end
                        bw_forehead=imfill(bw_forehead,'holes');
                        
                        center_r=round(pos_hand(2)+(pos_hand(4)/2));
                        center_c=round(pos_hand(1)+(pos_hand(3)/2));
                        th=0:0.1:360;
                        X_lefthand=round(center_r + (R*2*sind(th)));
                        Y_lefthand=round(center_c + (R*2*cosd(th)));
                        bw_lefthand(1:size(video_frame,1),1:size(video_frame,2))=0;
                        bw_lefthand=logical(bw_lefthand);
                        for i=1:length(X_lefthand)
                            bw_lefthand(X_lefthand(i),Y_lefthand(i))=1;
                        end
                        bw_lefthand=imfill(bw_lefthand,'holes');
                        
                        W=2;
                        [r,c]=find(bw_forehead==1);
                        forehead_G=video_frame(min(r):max(r),min(c):max(c),2);
                        bw_forehead_crop=bw_forehead(min(r):max(r),min(c):max(c));
                        mean_back=mean(forehead_G(bw_forehead_crop==1));
                        forehead_G(bw_forehead_crop==0)=round(mean_back);
                        [X1,Y1] = meshgrid(1:size(forehead_G,1),1:size(forehead_G,2));
                        [r,c]=find(forehead_G==max(forehead_G(:)));
                        start_r=round(mean(r)-W);
                        end_r=round(mean(r)+W);
                        start_c=round(mean(c)-W);
                        end_c=round(mean(c)+W);
                        if(start_r<1)
                            start_r=1;
                        end
                        if(start_c<1)
                            start_c=1;
                        end
                        if(end_r>size(forehead_G,1))
                            end_r=size(forehead_G,1);
                        end
                        if(end_c>size(forehead_G,2))
                            end_c=size(forehead_G,2);
                        end
                        G_mean_forehead(frame_no,1)=max(max(forehead_G(start_r:end_r,start_c:end_c)));
                        
                        [r,c]=find(bw_lefthand==1);
                        hand_G=video_frame(min(r):max(r),min(c):max(c),2);
                        bw_lefthand_crop=bw_lefthand(min(r):max(r),min(c):max(c));
                        mean_back=mean(hand_G(bw_lefthand_crop==1));
                        hand_G(bw_lefthand_crop==0)=round(mean_back);
                        [X2,Y2] = meshgrid(1:size(hand_G,1),1:size(hand_G,2));
                        [r,c]=find(hand_G==max(hand_G(:)));
                        start_r=round(mean(r)-W);
                        end_r=round(mean(r)+W);
                        start_c=round(mean(c)-W);
                        end_c=round(mean(c)+W);
                        if(start_r<1)
                            start_r=1;
                        end
                        if(start_c<1)
                            start_c=1;
                        end
                        if(end_r>size(hand_G,1))
                            end_r=size(hand_G,1);
                        end
                        if(end_c>size(hand_G,2))
                            end_c=size(hand_G,2);
                        end
                        G_mean_hand(frame_no,1)=max(max(hand_G(start_r:end_r,start_c:end_c)));

                        %////////////////////////////////////////////////////////////////////

                        figure(4)
                        subplot(2,2,1)
                        imshow(video_frame)
                        drawnow
                        hold on
                        rectangle('Position',box_face(1,:),'Curvature',[0.1,0.5],'EdgeColor','r')
                        hold on
                        rectangle('Position',pos_hand,'Curvature',[0.1,0.5],'EdgeColor','g')
                        hold on
                        plot(Y_forehead,X_forehead,'.b')
                        hold on
                        plot(Y_lefthand,X_lefthand,'.b')
                        hold on
                        title(strcat('Time:[',num2str(time_yet),'] sec'));
                        hold off
                        drawnow
                        
                        subplot(2,2,3)
                        surf(X1,Y1,forehead_G)
                        title('forehead ROI')
                        subplot(2,2,4)
                        surf(X2,Y2,hand_G)
                        title('hand ROI')
                        drawnow
                        
                        I=getframe(gcf);
                        writeVideo(v1,I)
                        
                        last_time=rr_time(1);
                        
    else
                       
                        G_mean_forehead(frame_no,1)=0;
                        G_mean_hand(frame_no,1)=0;
                        
                        last_time=rr_time(1);
                        
                        title(strcat('Time:',num2str(time_yet),'sec'));
                        drawnow
        
    end
     frame_no=frame_no+1;
end
close(v1)

%////////// applying wavelets and filtering //////////////

Fs = frame_rate; 
t = linspace(0,1,Fs); 

[r,~]=find(G_mean_forehead>0);
G_mean_forehead1=G_mean_forehead(min(r):max(r));

[r,c]=find(G_mean_hand>0);
G_mean_hand1=G_mean_hand(min(r):max(r));

G_forehead_flt = bpfilt(G_mean_forehead1,0.2,1,Fs);
G_hand_flt = bpfilt(G_mean_hand1,0.2,1,Fs);

[pks_forehead,locs_forehead] = findpeaks(G_forehead_flt,'MinPeakDistance',Fs/2);
[pks_hand,locs_hand] = findpeaks(G_hand_flt,'MinPeakDistance',Fs/2);

figure(21)
subplot(2,1,1)
plot(1:length(G_forehead_flt),G_forehead_flt,'-b',locs_forehead,pks_forehead,'or')
title('forehead filtered signal')
subplot(2,1,2)
plot(1:length(G_hand_flt),G_hand_flt,'-b',locs_hand,pks_hand,'or')
title('hand filtered signal')
drawnow

%////////////// predicting PTT using forehead and hand signals ///
k=1;
for i=1:length(pks_forehead)
    curr_forehead_loc=locs_forehead(i);
    [r,c]=find(locs_hand>curr_forehead_loc);
    if(~isempty(r))
        curr_hand_loc=locs_hand(r(1));
        A=(curr_hand_loc - curr_forehead_loc)/(Fs*2);
        if(A > 0.3 && A < 1)
            PTT_camera(k)=A;
            k=k+1;
        end
    else
        break
    end
end

figure(22)
bar(PTT_camera,0.7)
xlabel('instances')
ylabel('PTT (delay in forehead and hand peak)')
drawnow







