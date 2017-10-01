#!/bin/bash
####################################################################################################
# This script starts the process of applying the Benson et al. (2014) template to a freesurfer
# subject; it is intended for use with the apply_template.sh script in Docker.

if [ "$1" = "license" ] || [ "$1" = "LICENSE" ] || [ "$1" = "License" ]
then cat /LICENSE.txt
     exit 0
elif [ "$1" = "readme" ] || [ "$1" = "README" ] || [ "$1" = "Readme" ]
then cat /README.md
     exit 0
fi

if [ ! -d "/input" ] || [ ! -d "/input/surf" ] || [ ! -d "/input/mri" ]
then echo "SYNTAX:"
     echo "docker run -ti --rm -v /freesurfer/subject/dir:/input nben/occipital_atlas"
     echo " OR docker run -ti --rm license"
     echo " OR docker run -ti --rm readme"
     exit 1
fi

# Make sure our pythonpath is setup
export PYTHONPATH="$PYTHONPATH:/opt/neuropythy"

ln -s /input /opt/freesurfer/subjects/input || {
    echo "Could not link /input to /opt/freesurfer/subejcts/input"
    exit 1
}

# okay, we can now apply the templates normally
/opt/share/retinotopy-template/apply_template.sh input || {
  echo "apply_template.sh failed!"
  exit 1
}

#cp "$SUBJECTS_DIR/$SUBJID"/surf/?h.template_*.mgz /input/surf/
#cp "$SUBJECTS_DIR/$SUBJID"/surf/?h.wang2015_atlas.mgz /input/surf/
#cp "$SUBJECTS_DIR/$SUBJID"/mri/native.template_*.mgz /input/mri/
#cp "$SUBJECTS_DIR/$SUBJID"/mri/scanner.template_*.mgz /input/mri/
#cp "$SUBJECTS_DIR/$SUBJID"/mri/native.wang2015_atlas.mgz /input/mri/
#cp "$SUBJECTS_DIR/$SUBJID"/mri/scanner.wang2015_atlas.mgz /input/mri/

#rm -rf "$SUBJECTS_DIR/$SUBJID/"

exit 0
