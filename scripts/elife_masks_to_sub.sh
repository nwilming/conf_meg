#
# Convert eLife masks from JWdeGees eLife paper to subject specific masks.
#
printf -v SUBJECT "S%02d" $1 
subject_dir=/home/nwilming/fs_subject_dir/$SUBJECT

for label in `ls /home/nwilming/conf_analysis/required/masks/*.label`
do
    echo $label
    labelbase=`basename $label`
    output=JWDG_$labelbase
    labelname=${output/nii.gz/label}
    my_string=abc
    substring=ab
    if [ "${labelbase/lh}" = "$labelbase" ] ; then
      # Left hemi
      echo '--->>> HEMI=RIGHT!'
      mri_label2label --srclabel $label \
                    --srcsubject fsaverage \
                    --trgsubject $SUBJECT \
                    --regmethod surface \
                    --trglabel $labelname --hemi rh
    else
      echo '--->>> HEMI=LEFT!'
      mri_label2label --srclabel $label \
                    --srcsubject fsaverage \
                    --trgsubject $SUBJECT \
                    --regmethod surface \
                    --trglabel $labelname --hemi lh
    fi
        
done
