#
# This script converts the surface data of Wang et al. (2015)'s atlas into Freesurfer/MNE's label format.  
# The script will be particularly useful for MEG users because it is a preferred format for MNE analysis.
# 
# In this script, we use "bert" as an example of subjectid in freesurfer directory.
# 
# Requirement: Freesurfer:
# https://surfer.nmr.mgh.harvard.edu/
# 
# This script has been tested in Ubuntu 14.04.2 LTS.
#
# For information about Docker tools to generate surface map of Wang's atlas into single brain space:
# https://hub.docker.com/r/nben/occipital_atlas/
# 
# Information of original Wang et al. (2015)'s atlas:
# Wang L, Mruczek RE, Arcaro MJ, Kastner S (2015) Probabilistic Maps of Visual Topography in Human Cortex. Cereb Cortex 25(10):3911-31. doi:10.1093/cercor/bhu277
#
# Author: Hiromasa Takemura, Center for Information and Neural Networks (CiNet), Japan <htakemur@nict.go.jp>
#

export roiname_array=(1 "V1v" "V1d" "V2v" "V2d" "V3v" "V3d" "hV4" "VO1" "VO2" "PHC1" "PHC2" \
    "TO2" "TO1" "LO2" "LO1" "V3B" "V3A" "IPS0" "IPS1" "IPS2" "IPS3" "IPS4" \
    "IPS5" "SPL1" "FEF")
export subjid=$1

for i in {1..25}
do
 mri_cor2label --i ${SUBJECTS_DIR}/${subjid}/surf/lh.wang2015_atlas.mgz\
               --id ${i} --l lh.wang2015atlas.${roiname_array[${i}]}.label\
               --surf ${subjid} lh inflated
 mri_cor2label --i ${SUBJECTS_DIR}/${subjid}/surf/rh.wang2015_atlas.mgz\
               --id ${i} --l rh.wang2015atlas.${roiname_array[${i}]}.label\
               --surf ${subjid} rh inflated
done

# Generate eccentricity label
mri_binarize --i ${SUBJECTS_DIR}/${subjid}/surf/lh.template_eccen.mgz\
    --min 0.0001 --max 11 --o ${SUBJECTS_DIR}/${subjid}/surf/lh.eccen11_vol.mgz
mri_binarize --i ${SUBJECTS_DIR}/${subjid}/surf/rh.template_eccen.mgz\
    --min 0.0001 --max 11 --o ${SUBJECTS_DIR}/${subjid}/surf/rh.eccen11_vol.mgz
mri_vol2label --i ${SUBJECTS_DIR}/${subjid}/surf/lh.eccen11_vol.mgz\
    --id 1 --l lh.eccen11.label --surf ${subjid} lh inflated
mri_vol2label --i ${SUBJECTS_DIR}/${subjid}/surf/rh.eccen11_vol.mgz\
    --id 1 --l rh.eccen11.label --surf ${subjid} rh inflated