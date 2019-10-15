close all;
clear;clc

% get the ground truth masks
corTst = csvread('/path/labels_tst.csv', 1, 0);
corTst = sortrows(corTst);
corTst = corTst(:,1:5);

mskTst_eq = single(zeros(a ,b, c));

for i = 1:c
    idx = find(corTst(:,1) == i);
    d = length(idx);
    mskStack = single(zeros(a, b, d));
    for j = 1:d
        xmin = corTst(idx(j), 2)+0.5;
        xmax = corTst(idx(j), 3)+0.5;
        ymin = corTst(idx(j), 4)+0.5;
        ymax = corTst(idx(j), 5)+0.5;
        
        xi = [xmin, xmax, xmax, xmin, xmin];
        yi = [ymin, ymin, ymax, ymax, ymin];
        
        mskStack(:,:,j) = poly2mask(xi, yi, a, b);
    end
    
    mskTst_eq(:,:,i) = sum(mskStack, 3);
end

% get the ground truth volumes
ccA = bwconncomp(mskTst_eq);
statA = regionprops(ccA);
numObjA = ccA.NumObjects;

pixelIdxA = cell(1, numObjA);    % store the pixel index for comparison
volumeA = zeros(1, numObjA);
centroidAz = zeros(1, numObjA);
metSize_eq = zeros(1, numObjA);

for i = 1:numObjA
    pixelIdxA{i} = ccA.PixelIdxList{i};
    
    volumeA(i) = statA(i).Area;
    centroidAz(i) = statA(i).Centroid(3);
    
    xsize = statA(i).BoundingBox(4) * 0.9375;
    ysize = statA(i).BoundingBox(5) * 0.9375;
    zsize = statA(i).BoundingBox(6) * 1;
    
    metSize_eq(i) = max([xsize, ysize]);
end

figure;
edges = 0:3:60;
htst = histogram(metSize_eq, 'BinEdges', edges);
xticks(edges)
ylim([0, 100])
xlabel('met size (mm)')
ylabel('count')
title('test data met size distribution')
tstValues = htst.Values;

sum(metSize_eq < 3)
sum((metSize_eq >= 3) & (metSize_eq < 6))
sum(metSize_eq >= 6)


% generate the prediction masks
corPrd = csvread('/path/SSD_conf50.csv');
corPrd = sortrows(corPrd);

mskPrd = single(zeros(a, b, c));
for i = 1:c
    idx = find(corPrd(:,1) == i);
    d = length(idx);
    mskStack = single(zeros(a, b, d));
   for j = 1:d
        xmin = floor(corPrd(idx(j), 2))+0.5;
        xmax = floor(corPrd(idx(j), 4))+0.5;
        ymin = floor(corPrd(idx(j), 3))+0.5;
        ymax = floor(corPrd(idx(j), 5))+0.5;
                
        xi = [xmin, xmax, xmax, xmin, xmin];
        yi = [ymin, ymin, ymax, ymax, ymin];
        
        mskStack(:,:,j) = poly2mask(xi, yi, a, b);
   end
    mskPrd(:,:,i) = sum(mskStack, 3); 
end

ccB = bwconncomp(mskPrd);
statB = regionprops(ccB);
numObjB = ccB.NumObjects;

pixelIdxB = cell(1, numObjB);    % store the pixel index for comparison
volumeB = zeros(1, numObjB);
centroidBz = zeros(1, numObjB);
metSizePrd = zeros(1, numObjB);

centroidB = zeros(numObjB, 3);

for i = 1:numObjB
    pixelIdxB{i} = ccB.PixelIdxList{i};
    
    volumeB(i) = statB(i).Area;
    centroidBz(i) = round(statB(i).Centroid(3));
    
    centroidB(i,:) = round(statB(i).Centroid);
    
    xsize = statB(i).BoundingBox(4) * 0.9375;
    ysize = statB(i).BoundingBox(5) * 0.9375;
    zsize = statB(i).BoundingBox(6) * 1;
    
    metSizePrd(i) = max([xsize, ysize]);
end

figure;
edges = 0:3:60;
histogram(metSizePrd, 'BinEdges', edges);
xticks(edges)
xlabel('met size (mm)')
ylabel('count')
title('SSD all detection')



% compare the ground truth and prediction detection volumes
tpMets = zeros(1, numObjA);
fpMets = zeros(1, numObjB);

for i = 1:numObjB
    ctdB = centroidBz(i);
    pxlIdxB = pixelIdxB{i};
    tfpSwitch = 0;

    for j = 1:numObjA
        ctdA = centroidAz(j);
        
        if abs(ctdB - ctdA) <= 100
            pxlIdxA = pixelIdxA{j};        
            cmnPxl = intersect(pxlIdxB, pxlIdxA);
            
            if length(cmnPxl) >= 1
                tpMets(j) = metSize_eq(j);
                tfpSwitch = 1;
            end
        end
    end
    
    if tfpSwitch == 0
        fpMets(i) = metSizePrd(i);
    end
end

tpMets = tpMets(tpMets > 0);
fpMets = fpMets(fpMets > 0);

% histogram of detected mets size
figure;
edges = 0:3:60;
htp = histogram(tpMets, 'BinEdges', edges);
xticks(edges)
ylim([0, 100])
xlabel('met size of the max side (mm)')
ylabel('count')
title('SSD TP detection')
tpValues = htp.Values;

% overlay the detected mets histogram with ground truth
figure;
edges = 0:3:60;
histogram(metSize_eq, 'BinEdges', edges);
hold on
histogram(tpMets, 'BinEdges', edges, 'FaceAlpha', 0.5);
hold off
xticks(edges)
ylim([0, 100])
xlabel('met size (mm)')
ylabel('count')
title('SSD sensitivity overlay')
legend('gnd truth','detection')


% detect ratio
figure;
ratio = tpValues./tstValues;
bar(ratio)
ylim([0, 1.1])
xlabel('met size (mm)')
ylabel('ratio')
title('SSD sensitivity')

% split the met between 3 mm
tpMetsB3 = sum(tpMets < 3);
tpMetsA3B6 = sum((tpMets >= 3) & (tpMets < 6));
tpMetsA6 = sum(tpMets >= 6);

fpMetsB3 = sum(fpMets < 3);
fpMetsA3B6 = sum((fpMets >= 3) & (fpMets < 6));
fpMetsA6 = sum(fpMets >= 6);

