# Remote review with the native Rerun viewer

The review packages (`runs/*/*.rrd`, ~100 MB, H.264 video inside) are reviewed from another
computer over Tailscale. The Rerun **web** viewer loads over plain http but its video panes fail
with "failed to create video decoder": WebCodecs only exists in a secure (https) context. The
**native** viewer has no such limit, so the client installs it and connects to the server's gRPC
proxy.

**Server** (this repo, tailscaled running):

```bash
scripts/serve_review_over_tailscale.sh            # default: the v5 first-minute package
scripts/serve_review_over_tailscale.sh runs/<pkg>/<recording>.rrd --server-memory-limit 6GiB
```

It binds the Tailscale IPv4 from `tailscale ip -4`, hosts the web viewer on :9090 and the gRPC
proxy on :9876, and prints the connect command. Bound beyond loopback: no authentication, anyone
who can reach the address inside your tailnet can read the recording. Ctrl-C stops it.

**Client** (macOS/Linux: `install.sh`; Windows: `install.ps1`), then connect:

```bash
bash install.sh                                   # uv -> rerun-sdk==0.37.1 -> ffmpeg -> versions
bash connect.sh 100.64.0.7                     # = rerun --connect rerun+http://100.64.0.7:9876/proxy
```

Rules: the client's `rerun` must be **exactly the server's `rerun-sdk` version** (`RERUN_VERSION`
at the top of the installers mirrors `pyproject.toml`; bump both together). Rerun's native viewer
decodes H.264 through a separately installed **`ffmpeg` >= 5.1 on PATH** (Rerun's own docs:
https://rerun.io/docs/concepts/logging-and-ingestion/video); `rerun --version` must list `ffmpeg`
under "Video features". Browser-only client: `scripts/serve_review_over_tailscale.sh --https-port
8443` binds loopback and prints the `tailscale serve --bg --https=...` commands that give the web
viewer an https origin (untested here).
