close all; clear;clc

% get test patient slice range and indices
load('/raid/zzhou13/brainMets/datasets/detectPrep266/Rad_resub/DICOMs_and_props/patientDataTst.mat')
sliceNums = zeros(1,54);
sliceIdxs = cell(54,1);
for i = 1:54
    patient = patientDataTst(i).tumor_mask;
    sliceNums(i) = size(patient,3);
end

sliceIdxs{1} = 1:sliceNums(1);
for i = 2:54
    lastIdx = sliceIdxs{i-1};
    curStart = lastIdx(end) + 1;
    curEnd = lastIdx(end) + sliceNums(i);
    sliceIdxs{i} = curStart:curEnd;
end


% get the ground truth masks
corTst = csvread('/raid/zzhou13/brainMets/datasets/detectPrep266/Rad_resub/labels_tst_eql.csv', 1, 0);
corTst = sortrows(corTst);
corTst = corTst(:,1:5);

mskTst_eq = single(zeros(a,b,c));

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
ccEq = bwconncomp(mskTst_eq);
statEq = regionprops(ccEq);
numObjEq = ccEq.NumObjects;

pixelIdxEq = cell(1, numObjEq);    % store the pixel index for comparison
volumeEq = zeros(1, numObjEq);
centroidEqz = zeros(1, numObjEq);
metSize_eq = zeros(1, numObjEq);

for i = 1:numObjEq
    pixelIdxEq{i} = ccEq.PixelIdxList{i};
    
    volumeEq(i) = statEq(i).Area;
    centroidEqz(i) = statEq(i).Centroid(3);
    
    xsize = statEq(i).BoundingBox(4) * 0.9375;
    ysize = statEq(i).BoundingBox(5) * 0.9375;
    zsize = statEq(i).BoundingBox(6) * 1;
    
    metSize_eq(i) = max([xsize, ysize]);
end

sum(metSize_eq < 3)
sum((metSize_eq >= 3) & (metSize_eq < 6))
sum(metSize_eq >= 6)


% get ground truth volumes per patient
metsTotal = zeros(54,1);

ptMetSizeGT = cell(54,1);
ptMetPxIdGT = cell(54,1);

edges = 0:3:57;
ptMetGTbined = cell(54,1);

for i = 1:54
    patientMask = mskTst_eq(:,:,sliceIdxs{i});
    cc = bwconncomp(patientMask);
    stat = regionprops(cc);
    numObj = cc.NumObjects;
    
    metsTotal(i) = numObj;

    pixelIdx = cell(1, numObj);    % store the pixel index for comparison
    volume = zeros(1, numObj);
    centroid = zeros(1, numObj);
    metSize = zeros(1, numObj);

    for j = 1:numObj
        pixelIdx{j} = cc.PixelIdxList{j};
    
        volume(j) = stat(j).Area;
        centroid(j) = stat(j).Centroid(3);
    
        xsize = stat(j).BoundingBox(4) * 0.9375;
        ysize = stat(j).BoundingBox(5) * 0.9375;
        zsize = stat(j).BoundingBox(6) * 1;
    
        metSize(j) = max([xsize, ysize]);
    end
    
    ptMetSizeGT{i} = metSize;
    ptMetPxIdGT{i} = pixelIdx;
    
    hGT = histogram(metSize, 'BinEdges', edges);
    GTValues = hGT.Values;
    ptMetGTbined{i} = GTValues;
end
    

% generate the prediction masks
corPrd = csvread([pwd '/SSD_ResNet_conf50.csv']);
corPrd = sortrows(corPrd);

mskPrd = single(zeros(a,b,c));
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

% get the prediction volumes per patient
ptMetSizePD = cell(54,1);
ptMetPxIdPD = cell(54,1);

for i = 1:54
    patientMask = mskPrd(:,:,sliceIdxs{i});
    cc = bwconncomp(patientMask);
    stat = regionprops(cc);
    numObj = cc.NumObjects;

    pixelIdx = cell(1, numObj);    % store the pixel index for comparison
    volume = zeros(1, numObj);
    centroid = zeros(1, numObj);
    metSize = zeros(1, numObj);

    for j = 1:numObj
        pixelIdx{j} = cc.PixelIdxList{j};
    
        volume(j) = stat(j).Area;
        centroid(j) = stat(j).Centroid(3);
    
        xsize = stat(j).BoundingBox(4) * 0.9375;
        ysize = stat(j).BoundingBox(5) * 0.9375;
        zsize = stat(j).BoundingBox(6) * 1;
    
        metSize(j) = max([xsize, ysize]);
    end    
    ptMetSizePD{i} = metSize;
    ptMetPxIdPD{i} = pixelIdx;
end
    

% compare the ground truth and prediction volumes
tpMets = cell(54, 1);
fpMets = cell(54, 1);

edges = 0:3:57;
tpMetsBined = cell(54,1);
fpMetsBined = cell(54,1);

for k = 1:54
    ptGTmetSize = ptMetSizeGT{k};
    ptGTmetPixs = ptMetPxIdGT{k};
    
    ptPDmetSize = ptMetSizePD{k};
    ptPDmetPixs = ptMetPxIdPD{k};
    
    tpList = zeros(1,length(ptGTmetSize));
    fpList = zeros(1,length(ptPDmetSize));
    
    for i = 1:length(ptPDmetSize)
        pxlIdxB = ptPDmetPixs{i};
        tfpSwitch = 0;

        for j = 1:length(ptGTmetSize)
            pxlIdxA = ptGTmetPixs{j};        
            cmnPxl = intersect(pxlIdxB, pxlIdxA);
            
            if length(cmnPxl) >= 1
                tpList(j) = ptGTmetSize(j);
                tfpSwitch = 1;
            end
        end
  
        if tfpSwitch == 0
            fpList(i) = ptPDmetSize(i);
        end
    end
    
    tpList = tpList(tpList > 0);
    fpList = fpList(fpList > 0);
    
    tpMets{k} = tpList;
    fpMets{k} = fpList;
    
    
    htp = histogram(tpList, 'BinEdges', edges);
    tpValues = htp.Values;
    
    hfp = histogram(fpList, 'BinEdges', edges);
    fpValues = hfp.Values;
    
    tpMetsBined{k} = tpValues;
    fpMetsBined{k} = fpValues;
end


% integrate binned cells together
ptMetGTtable = cat(1, ptMetGTbined{:});
ptMetTPtable = cat(1, tpMetsBined{:});
ptMetFPtable = cat(1, fpMetsBined{:});

ptSen = ptMetTPtable./ptMetGTtable;
boxplot(ptSen,'Symbol','ko', ...
        'Labels', ...
        {'0-3','3-6','6-9','9-12','12-15','15-18','18-21', ...
         '21-24','24-27','27-30','30-33','33-36','36-39', ...
         '39-42','42-45','45-48','48-51','51-54','54-57'}, ...
         'Whisker', 1);

% overall sensitivity and precision
ptAllSen = sum(ptMetTPtable,2)./sum(ptMetGTtable,2);
sum(ptMetTPtable(:));

ptAllPrc = sum(ptMetTPtable,2)./(sum(ptMetTPtable,2) + sum(ptMetFPtable,2));
sum(ptMetFPtable(:));

% 0-3 sensitivity and precision
pt03Sen = ptMetTPtable(:,1)./ptMetGTtable(:,1);
sum(ptMetTPtable(:,1));

pt03Prc = ptMetTPtable(:,1)./(ptMetTPtable(:,1) + ptMetFPtable(:,1));
sum(ptMetFPtable(:,1));

% 3-6 sensitivity and precision
pt36Sen = ptMetTPtable(:,2)./ptMetGTtable(:,2);
sum(ptMetTPtable(:,2));

pt36Prc = ptMetTPtable(:,2)./(ptMetTPtable(:,2) + ptMetFPtable(:,2));
sum(ptMetFPtable(:,2));

% >6 sensitivity and precision
ptA6Sen = sum(ptMetTPtable(:,3:end),2)./sum(ptMetGTtable(:,3:end),2);
sum(sum(ptMetTPtable(:,3:end)));

ptA6Prc = sum(ptMetTPtable(:,3:end),2)./(sum(ptMetTPtable(:,3:end),2) + sum(ptMetFPtable(:,3:end),2));
sum(sum(ptMetFPtable(:,3:end)));


