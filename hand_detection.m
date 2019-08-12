function pos=hand_detection(img)
    
    img_ycbcr = rgb2ycbcr(img);
    Cr = img_ycbcr(:,:,3);
    
    img_b = Cr>=140 & Cr<=160;
    se=strel('disk',10);
    img_b1=imerode(img_b,se);
    img_b2 = imclearborder(img_b1);
    img_b3=bwareaopen(img_b2,10000);
    img_b3=imfill(img_b3,'holes');
    [lab,~]=bwlabel(img_b3);
    [r,c]=find(lab==1);
    
    pos=[min(c) min(r) max(c)-min(c) max(r)-min(r)];
%     figure
%     imshow(img_b3)
%     hold on
%     rectangle('Position',pos,'Curvature',[0.1,0.5],'EdgeColor','r')
%     hold off

end

