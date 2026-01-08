###Here, we used multi-region segmented MRIs obtained from nnU-Net to generate binary segmentations
### prepared for SERA, including the enhancing tumor, tumor core, and edema.
### By Aliye Hashemi
### Henry Ford Health System --- Hermelin Brain Tumor Center


#######################################  Enhancing Tumor (label 3 only)
import nibabel as nib
import numpy as np

# Load the NIfTI segmentation file
nifti_img = nib.load("C:/patient006.nii.gz")
segmentation = nifti_img.get_fdata()
binary_segmentation = np.zeros_like(segmentation, dtype=np.uint8)
binary_segmentation[segmentation == 4] = 1
binary_nifti = nib.Nifti1Image(binary_segmentation, affine=nifti_img.affine, header=nifti_img.header)
nib.save(binary_nifti, "C:/binary segmentation/patient006_label3.nii.gz")

print("Done!")



###################################### Tumor Core (labels 1&3)

# Load the NIfTI segmentation file
nifti_img = nib.load("C:/patient006.nii.gz")
segmentation = nifti_img.get_fdata()
binary_segmentation = np.zeros_like(segmentation, dtype=np.uint8)
binary_segmentation[(segmentation == 0) | (segmentation == 2)] = 0
binary_segmentation[(segmentation == 1) | (segmentation == 4)] = 1
binary_nifti = nib.Nifti1Image(binary_segmentation, affine=nifti_img.affine, header=nifti_img.header)
nib.save(binary_nifti, "C:/binary segmentation/patient006_labels31.nii.gz")

print("Done!")


######################################  Edema (label 2 only)

# Load the NIfTI segmentation file
nifti_img = nib.load("C:/patient006.nii.gz")
segmentation = nifti_img.get_fdata()
binary_segmentation = np.zeros_like(segmentation, dtype=np.uint8)
binary_segmentation[segmentation == 2] = 1
binary_nifti = nib.Nifti1Image(binary_segmentation, affine=nifti_img.affine, header=nifti_img.header)
nib.save(binary_nifti, "C:/binary segmentation/patient006_label2.nii.gz")

print("Done!")

