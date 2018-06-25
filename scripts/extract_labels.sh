printf -v SUBJECT "S%02d" $1 
outdir=/home/nwilming/fs_subject_dir/$SUBJECT/label/a2009slabels
labeldir=/home/nwilming/fs_subject_dir/$SUBJECT/label

mkdir -p $outdir
mri_annotation2label --annotation aparc.a2009s  --subject $SUBJECT --hemi lh --outdir $outdir
mri_annotation2label --annotation aparc.a2009s  --subject $SUBJECT --hemi rh --outdir $outdir


for label in `ls $outdir/*.label`
do
    base="`basename $label`"
    mv $label $labeldir/a2009s.$base 
done
