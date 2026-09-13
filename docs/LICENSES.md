# License and handling policy

This repository contains original code, configuration, and synthetic test coordinates.
It includes no third-party recordings, annotations, model weights, or copied
third-party source code. The ignored local `data/` directory contains approved G2
inputs outside Git.

## Noncommercial and provenance constraints

- An NC license can restrict job-seeking, demos, or other activity that confers a
  commercial advantage. This lab does not presume a private presentation is permitted.
- Dataset terms can add conditions beyond the headline license. Keep the current terms,
  access agreement, required citation, and display restrictions with each source entry.
- Do not redistribute raw videos, annotations, derived mask payloads, checkpoints, or
  gated assets through Git, Rerun recordings, screenshots, or release archives unless
  the applicable rights explicitly allow it.
- Rerun outputs from this session reference mask artifact URIs only; they never embed
  full mask payloads. Future exports must preserve this rule unless approved terms say
  otherwise.

## Current core-method inputs

- **Assembly101 clip `assembly101-nusar-9033-g2`:** the approved configuration records
  `CC BY-NC 4.0` and the local provenance ledger. G1 and G2 are user-approved for the
  exact raw recordings and their 180-second, audio-free proxies. This is provenance
  approval for the local smoke procedure, not a legal conclusion that a job-seeking,
  private demo, or public display is noncommercial. Raw video, proxy video, derived
  masks, annotations, and RRD files remain ignored and must not be redistributed.
- **MuggledSAM source:** `/home/nick/src/muggled_sam` declares Apache-2.0. Battle uses
  its public interface and does not copy its code. Apache-2.0 covers that source, not
  the SAM3 checkpoint.
- **SAM3/SAM3.1 checkpoint:** MuggledSAM's README states that SAM3 weights require an
  agreement before downloading. The smoke run used an already-local checkpoint; Battle
  did not download, bundle, or redistribute it. Its checkpoint license/access status
  must be reviewed separately before a successful model run can be shared.

## Dependency and future-model review

Python packages are pinned in `pyproject.toml` and resolved in `uv.lock`. Review their
current license notices before redistribution. Future adapters must separately document
the code license, checkpoint license, access requirements, and attribution for every
model; an open-source repository does not automatically grant rights to its weights.

## Decision record template

For each future source or model, record:

1. identifier and version;
2. authoritative terms URL and retrieval date;
3. intended use, audience, and duration;
4. attribution or citation required;
5. redistribution decision;
6. reviewer and approval status.
