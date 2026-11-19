# Registrations

Each file in this directory is the text that was frozen before the
corresponding analysis ran on confirmatory data.  The texts are reproduced
verbatim; the SHA256 of each one is in `HASHES.md` and in the decision log that
was kept during the work.

| file | data | frozen (UTC) | predictions |
|---|---|---|---|
| `preregistration_coco.md` | COCO val2017 holdout | before the COCO analyses | the COCO size-gap predictions |
| `registration_voc2007.md` | VOC 2007 test | 2026-11-16 11:05 | H1-H4 |
| `addendum_G.md` | VOC 2007 test | 2026-11-17 15:20 | G1-G2 |
| `addendum_N.md` | VOC 2007 test | 2026-11-18 02:05 | N1-N3 |
| `addendum_D3.md` | VOC 2007 test | 2026-11-18 06:20 | D1-D3 |
| `addendum_G5.md` | VOC 2007 test | 2026-11-18 08:22 | G5 |
| `addendum_R.md` | VOC 2007 test | 2026-11-18 12:22 | R1-R3 |
| `addendum_Y.md` | VOC 2007 test | 2026-11-18 17:28 | Y1 |
| `addendum_Y2.md` | VOC 2012 val | 2026-11-18 20:05 | Y2 |

Other files:

* `OUTCOMES.md` - all eighteen predictions with their estimates, intervals and
  outcomes.
* `TIMELINE.md` - freeze time, first run, first look, and what was known at
  freeze time, for every registration.
* `DEVIATIONS.md` - every departure from a registration or from the announced
  protocol, with the reason.
* `HASHES.md` - SHA256 of every registration and every frozen script, both the
  version recorded during the work and the released version.
