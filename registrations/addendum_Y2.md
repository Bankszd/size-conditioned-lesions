Addendum Y2 (VOC 2012 val): independent test of the model-comparison reduction
Rationale and timing: Y2 was initiated after the inconclusive VOC 2007 result (Y1) was known. VOC 2012 validation is the final dataset on which we will test this prediction in this revision, regardless of the outcome. Y1 and Y2 will be reported together with their point estimates and intervals.
Data: PASCAL VOC 2012 detection val, the 5,823 image IDs of ImageSets/Main/val.txt (ID list SHA256 recorded in the decision log); VOC->COCO class map; difficult objects as crowd; COCO size thresholds; content-hash overlap with project VOC 2007 images and project use of VOC 2011/2012 checked before freezing and reported.
Models, readout, depths, rules (four, prop50, geo50), estimands D, I, T = I_four - I_prop50 and T_g = I_geo50 - I_prop50: exactly as in Addendum Y.
Statistics: one shared 2,000 x 5,823 whole-image resample matrix, PCG64 seed 20271001, duplicates counted k times as in Addendum Y; two-sided 95% percentile intervals.
Known before freezing: all VOC 2007 test results including Y1 (T = +4.24 [-0.10, +14.04], inconclusive); all COCO results. Not known: any VOC 2012 model output.
Sole primary criterion: Y2 is confirmed if the lower bound of the two-sided 95% percentile interval of T exceeds 0; contradicted if the upper bound is below 0; inconclusive otherwise.
Y1 remains inconclusive whatever Y2 shows.
Descriptive, no prediction, whatever its value: T_g; I_four, I_prop50, I_geo50; AP, AP_S, AP_L and their changes for every arm; number of non-difficult small GT objects evaluated.
All outcomes are reported.
