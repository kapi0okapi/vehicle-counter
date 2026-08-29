# Getting UA-DETRAC

UA-DETRAC (Wen et al., CVIU 2020) is 10 hours of traffic-camera video (25 fps, 960x540) from 24
sites in Beijing and Tianjin, with ~1.2 M annotated boxes and track IDs.

1. Download the training images (`DETRAC-train-data.zip`, folder `Insight-MVT_Annotation_Train`)
   and the XML annotations (`DETRAC-Train-Annotations-XML.zip`). The original site
   (detrac-db.rit.albany.edu) has moved several times; current mirrors are listed on
   Papers with Code ("UA-DETRAC") and Kaggle. Check the licence of the mirror you use.
2. Copy both folders to the cluster, e.g. `/cluster/work/$USER/vehicle-counter/data/ua_detrac/`.
3. Convert (CPU job): `python3 slurm/generate.py convert-detrac --submit`.

Any other dataset works too: the counting pipeline only needs a YOLO-format `data.yaml` for
training and, for count evaluation, folders with `video.mp4` + MOT-format `gt.txt`.
