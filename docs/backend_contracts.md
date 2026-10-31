# Backend contracts — stable reference path

Backends provide typed execution, monitoring, checkpoint/resume and failure evidence for the control plane. They should return **lightweight artifact references**, optional verified hashes and status; never embed model weights, raw private datasets or large logs into persisted control records.

Provider allocation is governed by explicit [compute policy](compute_execution_policy.md) and [safe operations](operations/run-safety.md). [Data and state architecture](architecture/data-and-state.md) and [durable model storage design](durable_model_storage_and_repository_evidence.md) explain artifact custody and retrieval. A historical backend's existence in Git does not make its previous paid-run approval reusable.
