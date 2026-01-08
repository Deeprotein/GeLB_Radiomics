niftiFolder = 'C:\nifti';
files = dir(fullfile(niftiFolder, '*.nii.gz'));

if isempty(files)
    error('No .nii.gz files found');
end
filePath = fullfile(niftiFolder, files(1).name);
vol = niftiread(filePath);

vol_vals = double(vol);

save(fullfile(niftiFolder, 'PETimg.mat'), 'vol_vals');
fprintf('Saved 3D volume from file %s\n', files(1).name);
