% Load the ROI from NIfTI (.nii or .nii.gz)
roi_nifti = niftiread('C:\Aliye\HFHS\MRI\Pilot\Correct Files\patient 017\09062019\binary segmentation\patient017_label2.nii.gz'); 

total = roi_nifti;

output_path = 'C:\Aliye\HFHS\MRI\SERA\Aliye\nifti\Recurrence FU\62875676_17\09062019\2\contours.mat';
save(output_path, 'total');
