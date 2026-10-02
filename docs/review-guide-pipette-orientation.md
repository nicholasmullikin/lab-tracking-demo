# Pipette orientation review

This review connects camera masks to a blue pipette's fitted 3D shaft and dispensing end.
It shows all four pipettes in 2D and the blue pipette in 3D. Read the camera evidence beside
the arrow: a resolved estimate can still follow a contaminated or incomplete shaft.

## Five-minute route

Start with the **Orientation** tab. The blue segment is fitted visible extent in centimeters;
the magenta arrow and white endpoint identify the estimated dispensing end. Green camera
segments show the same fit reprojected into each image. An unresolved direction has no arrow.

1. At raw 0, compare the resting blue shaft and dispensing end across cameras.
2. At raw 100, follow pickup. Check that blue follows the held pipette while the other slots stay put.
3. At raw 200, inspect glove leakage and the direction abstention in the reviewed baseline.
4. Move to the later reviewed frames listed in the **Evidence** tab. Compare the selected policy
   with the direct inspection notes; a fixed slot does not prove physical identity.
5. At the final recorded frame, check what remains visible, what resolves and what fails.

The **Cameras** tab shows all six uncropped images with blue, yellow, red and multichannel
masks and axes. The **Evidence** tab gives cue contributions, camera support, consistency
and frame-specific review notes. Dropped-camera residuals retain original pixel units.
The overview samples camera images twice per second and preserves native-rate world geometry.
Use a saved native clip when inspecting motion between overview images.
Cue signs follow the fitter's endpoint order. A sign change alone does not prove a physical flip.

## Saved bundle

The saved demo lives under `runs/pipette-demo-20261002-final/`, outside Git.
It freezes raw 0–899: 900 native frames, 30.03 seconds, baseline masks and instantaneous
color/taper direction. The longer extension, policy comparisons and temporal orientation
comparison remain unfinished and are excluded from this bundle.
Its archive includes native and HTTPS browser verification screenshots and reports.
Its manifest freezes the source policy, reviewed stage, orientation method, frame range,
review route and hashes. Earlier previews have separate names and are not final results.

```bash
.venv/bin/rerun --memory-limit 8GiB runs/pipette-demo-20261002-final/overview.rrd runs/pipette-demo-20261002-final/overview-demo.rbl
```

Separate orientation, cameras and evidence layouts are saved alongside the recording.
The native recordings name their exact raw-frame intervals. Images and masks are embedded;
opening the bundle needs no original videos, mask directories or model environment.

The matching `.tar.gz` archive and archive checksum sit beside the bundle. Restore into an
empty directory and run `sha256sum -c SHA256SUMS` there. Rebuilding needs the original videos
and masks named in the saved configuration and geometry. No inference is needed to export.
The archive is local to this machine, not an off-machine backup.

## Private browser review

```bash
bash scripts/serve_pipette_demo.sh --dry-run
bash scripts/serve_pipette_demo.sh --background
```

The helper binds loopback, serves the viewer on 9092 and the proxy on 9878, and prints the
Tailscale HTTPS setup and browser URL. These ports are separate from the experiment viewer.
The machine must be logged into Tailscale and remain online. Reviewers need access to its tailnet.
Use the printed DNS name in these commands:

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:9092
tailscale serve --bg --https=8444 http://127.0.0.1:9878
```

The helper prints a browser URL with WebGL rendering, a dark theme and fresh viewer state.
Its source is `rerun+https://HOST:8444/proxy`; use the printed URL to preserve proper encoding.
The native alternative is `rerun rerun+https://HOST:8444/proxy` with Rerun 0.37.1.
Stop the foreground helper with Ctrl-C; restart with the same bundle and command.
To keep serving after the terminal closes, use `bash scripts/serve_pipette_demo.sh --background`.
This starts the `battle-pipette-demo` user service. Inspect it with
`systemctl --user status battle-pipette-demo` and stop it with
`systemctl --user stop battle-pipette-demo`. Start it again after a machine reboot.
Remove only these HTTPS routes with `tailscale serve --https=8443 off` and
`tailscale serve --https=8444 off`. Preserve other serving routes.

## Public browser review

The public website is a separate, read-only presentation under
`runs/pipette-demo-20261002-public/`. It loads the saved 30-second overview and its
layout automatically. Its labels name each pipette explicitly. The top bar provides
**3D + cameras**, **All cameras** and **Evidence and review notes** views, a color
legend and a **Show masks** checkbox. Hiding masks retains pipette labels, shaft lines
and the current frame. Green lines are the 3D fit projected into each camera;
magenta identifies the estimated dispensing end.
Anyone with the public link can view and download that recording;
reviewers need no Tailscale account. The machine must remain online.

`scripts/serve_pipette_demo_public.py` binds to loopback on 9093 and serves only the
viewer assets, `overview.rrd` and the allowlisted layout files. It rejects writes, directory
listing, unpublished files and symlinks. The live Rerun proxy remains private.
The overview is re-exported from the frozen checkpoint to improve presentation;
the original archive and its checksums remain unchanged.

The public copy includes the installed Rerun 0.37.1 web assets, with compressed JavaScript
and WebAssembly. Its index defaults to the saved recording and layout, WebGL, a dark
theme and fresh state. The toolbar source is
`scripts/pipette_demo_public_controls.html`; the exporter’s `public_controls=True`
option generates the six mask/view layouts. Controls load only a small layout file,
with no recording reload. Local provenance is saved in `source-manifest.json`, which the
public server does not serve.

```bash
systemd-run --user --unit battle-pipette-demo-public --collect \
  --property Restart=on-failure --property RestartSec=3 \
  "$PWD/.venv/bin/python" "$PWD/scripts/serve_pipette_demo_public.py" \
  "$PWD/runs/pipette-demo-20261002-public"
tailscale funnel --bg --https=10000 http://127.0.0.1:9093
```

Use the public HTTPS URL printed by Funnel. Inspect with
`tailscale funnel status` and `systemctl --user status battle-pipette-demo-public`.
Disable just this public endpoint with `tailscale funnel --https=10000 off`, then
`systemctl --user stop battle-pipette-demo-public`. Other routes remain available.
Restart the service after a machine reboot.

## Interpretation

The older temporal orientation vote was measured against frozen manual tip clicks.
The prompt study used exploratory visual grades, the line comparison measured geometric
consistency, and the new multiplex run uses sparse direct inspection. Their numbers describe
different protocols. Camera agreement, resolved direction and stable slots are not accuracy.

See [recent results](results.md#pipette-prompt-and-line-study), the
[full-video checkpoint](qa/pipette-full-video-checkpoint-20261002/README.md) and the
[multiplex checkpoint](qa/pipette-multiplex-20261002/README.md).
FineBio frames and derived outputs retain the attribution in [LICENSES.md](LICENSES.md).
