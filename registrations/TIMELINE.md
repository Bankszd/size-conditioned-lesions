# Timeline of the registered analyses

All times are UTC.  "Known at freeze" lists the results that already existed
when the text was hashed; it is the only thing that distinguishes a prediction
from a post-hoc description.

## CONFIRM_REG_v1 (H1-H4, VOC 2007 test)

| event | time |
|---|---|
| frozen and hashed | 2026-11-16 11:05 |
| first VOC forward pass | 2026-11-16, after the hash |
| H1-H4 complete | before 2026-11-17 15:35 |
| known at freeze | all COCO results; no VOC result of any kind |

## Addendum G (G1-G2, VOC 2007 test)

| event | time |
|---|---|
| frozen and hashed | 2026-11-17 15:20 |
| first shard launched | 2026-11-17 15:21 |
| user authorisation ("start") | 2026-11-17 15:36 |
| known at freeze | all COCO results, including the Task G development run |

Process deviation: the two shards were launched at 15:21, fifteen minutes
before the explicit authorisation at 15:36, because the addendum had already
been supplied, hashed and written to the decision log.  The launch is recorded
here and in `DEVIATIONS.md`.  No Task G statistic on a VOC image existed at hash
time, and no output of the 15:21 launch was inspected before the authorisation.

## Addendum N (N1-N3, VOC 2007 test)

| event | time |
|---|---|
| frozen and hashed | 2026-11-18 02:05 |
| first VOC forward pass | 2026-11-18 02:05, after the hash |
| complete | 2026-11-18 10:40 |
| known at freeze | COCO development runs of Task N |

The run used a decoder configuration that turned out not to match the
registered one; it was re-executed under the registered configuration at
15:28 and only that re-execution carries the verdict.  See `DEVIATIONS.md`.

## Addendum D3 (D1-D3, VOC 2007 test)

| event | time |
|---|---|
| frozen and hashed | 2026-11-18 06:20 |
| first VOC forward pass | 2026-11-18 06:20, after the hash |
| complete | 2026-11-18 06:43 |
| known at freeze | the COCO development run of Task D3 |

## Addendum G5 (G5, VOC 2007 test)

| event | time |
|---|---|
| frozen and hashed | 2026-11-18 08:22 |
| DETR shards cleared to start | 2026-11-18 08:22 |
| aborted launch, no result used | 2026-11-18 08:26 |
| RT-DETRv2 code frozen | 2026-11-18 08:45 |
| complete | 2026-11-18 11:20 |
| known at freeze | all COCO results, including the G4 development run whose success criterion was not met |

## Addendum R (R1-R3, VOC 2007 test)

| event | time |
|---|---|
| frozen and hashed | 2026-11-18 12:22 |
| first VOC forward pass | 2026-11-18 12:22, after the hash |
| complete | 2026-11-18 13:05 |
| known at freeze | the COCO analyses R0 and R1, and the already-registered VOC four-cell gap |

## Addendum Y (Y1, VOC 2007 test)

| event | time |
|---|---|
| frozen and hashed | 2026-11-18 17:28 |
| geo50 implementation frozen | 2026-11-18 17:52 |
| first VOC forward pass of any Y arm | 2026-11-18, after 17:52 |
| known at freeze | all COCO results; the VOC four-cell size gap from Task R |

## Addendum Y2 (Y2, VOC 2012 val)

| event | time |
|---|---|
| frozen and hashed | 2026-11-18 20:05 |
| first VOC 2012 forward pass | after the hash |
| known at freeze | every VOC 2007 test result, including Y1; all COCO results |

Y2 is the last dataset on which this prediction is tested in the revision.  Y1
stays inconclusive whatever Y2 shows.
