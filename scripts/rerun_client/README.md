# Remote review with the native Rerun viewer

I open the review packages (`runs/*/*.rrd`, about 100 MB, with H.264 video inside) from another
computer over Tailscale. The Rerun web viewer loads over plain http, but its video panes fail
with "failed to create video decoder": WebCodecs exists only in a secure https context. The
native viewer has no such limit, so the client installs it and connects to the server's gRPC
proxy.

On the server, which is this repo with tailscaled running:

```bash
scripts/serve_review_over_tailscale.sh            # default: v6 interaction_review_combined.rrd + segmentation.rbl
scripts/serve_review_over_tailscale.sh runs/<pkg>/<recording>.rrd [runs/<pkg>/<preset>.rbl] --server-memory-limit 6GiB
```

It binds the Tailscale IPv4 from `tailscale ip -4`, hosts the web viewer on :9090 and the gRPC
proxy on :9876, and prints the connect command. Because it binds beyond loopback with no
authentication, anyone who can reach the address inside the tailnet can read the recording.
Ctrl-C stops it.

On the client, run `install.sh` on macOS or Linux and `install.ps1` on Windows, then connect:

```bash
bash install.sh                                   # uv -> rerun-sdk==0.37.1 -> ffmpeg -> versions
bash connect.sh host.example.ts.net               # replace with your server's host or IP
```

Use the server address printed by the serve script. The installers can run without a host;
the connection scripts require one.

Rules: the client's `rerun` must be exactly the server's `rerun-sdk` version. `RERUN_VERSION` at
the top of the installers mirrors `pyproject.toml`, so bump both together. Rerun's native viewer
decodes H.264 through a separately installed `ffmpeg` 5.1 or later on PATH (Rerun's own docs:
https://rerun.io/docs/concepts/logging-and-ingestion/video), and `rerun --version` must list
`ffmpeg` under "Video features". For a browser-only client,
`scripts/serve_review_over_tailscale.sh --https-port 8443` binds loopback and prints the
`tailscale serve --bg --https=...` commands that give the web viewer an https origin. I have not
tested that path.
