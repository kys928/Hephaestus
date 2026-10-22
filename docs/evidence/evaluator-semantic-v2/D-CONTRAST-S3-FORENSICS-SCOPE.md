# Read-only S3 forensics scope — failed D contrast pod

Run ID: `D-contrast-39299580607`; pod `lw476giazga8e4`; [GitHub Actions #39299580607](https://github.com/kys928/Hephaestus/actions/runs/39299580607).

The source run timed out in the launcher at about 940 seconds, with an estimated $0.6501 GPU charge, and its pod was verified deleted on the first attempt. Scientific inference is **not** proven finished and the independent audit job was skipped.

This branch performs **CPU-only read-only S3 forensics** using `ListObjectsV2`, `HeadObject` and `GetObject` on the single archived run prefix; it does not allocate a pod or train, repair, change scientific gates, mutate original S3 objects, issue an approval, or replay the consumed experiment. It checks:

- last persisted container bootstrap stage and timestamp (older stage history overwritten);
- whether the post-load model manifest exists and matches the frozen D identity;
- progress file and saved sample shard inventory, independently hash-checked with S3 receipt timestamps;
- known 56-prompt identity mapping, partial JSON parse formats and sum of saved generation latencies;
- number of complete observed contrast pairs and descriptive partial paired outcomes, explicitly NOT extrapolated to full 60-pair scientific accuracy;
- absence/presence of terminal result, S3 one-use claim, and approximate time allocation among bootstrapping/model load and persisted inference, with clear uncertainty.

An isolated `.[s3,test]` GitHub Actions job writes **only a local report artifact** in Actions, not the RunPod S3 bucket. No secrets, GPU UUIDs or raw model text appear in the job logs. If S3 has no manifest or saved shards, this is evidence of missing durable records, not proof that no transient inference ever happened.

**Do not merge or restart the experiment as part of this diagnostic.**
