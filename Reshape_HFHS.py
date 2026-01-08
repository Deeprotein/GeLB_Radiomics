
### Here, we aim to unify all four MRI sequences (T1, T1c, T2, and F) in terms of shape to prepare them for segmentation.
### By Aliye Hashemi
### Henry Ford Health System --- Hermelin Brain Tumor Center


import SimpleITK as sitk
import os

### Input 4 MRI sequences
input_dir = r"C:\input_images"
patient_id = "Patient006"

modalities = [
    f"{patient_id}_0000.nii.gz",  # T1
    f"{patient_id}_0001.nii.gz",  # T1c
    f"{patient_id}_0002.nii.gz",  # T2
    f"{patient_id}_0003.nii.gz",  # FLAIR
]

### Load the reference image to resample others to match this one
ref_image = sitk.ReadImage(os.path.join(input_dir, modalities[0]))

for modality in modalities:
    img_path = os.path.join(input_dir, modality)
    image = sitk.ReadImage(img_path)

    if image.GetSize() != ref_image.GetSize():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampled_img = resampler.Execute(image)
        sitk.WriteImage(resampled_img, img_path)
    else:
        print(f"{modality} already matches shape")

print("Done!")
