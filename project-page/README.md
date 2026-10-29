# Project page

Static site for **Preconditioned Flow Matching** (NeurIPS 2026), plus the code that
renders its concept video. It lives in the same repository as the released
implementation and has no build step: plain HTML, CSS and one small JS file.

```
project-page/
  index.html            the page
  styles.css            design system and layout
  main.js               scroll spy, autoplay, lightbox, copy-bibtex
  assets/
    figures/            paper figures, resized for the web
    video/              pfm_concept.mp4 + poster
  build/
    make_assets.py      trains the 2D models behind the video (reuses ../pfm)
    make_video.py       renders the frames and encodes the mp4
    make_web_assets.py  resizes/crops the paper figures into assets/figures
```

## Preview locally

```bash
cd project-page
python -m http.server 8000      # then open http://localhost:8000
```

`python -m http.server` does not answer Range requests, so the browser cannot seek
inside the mp4. The player detects this (`Accept-Ranges` missing) and loads the video
as a Blob instead, which makes scrubbing work locally; on any normal host — GitHub
Pages included — it streams and seeks natively. If you prefer a range-capable preview:
`npx http-server -c-1`.

## Publish

The site is self-contained, so any static host works. For GitHub Pages:

1. Push the repository.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. The workflow in `.github/workflows/pages.yml` publishes this folder on every
   push to `main`, at `https://<user>.github.io/<repo>/`.

Alternatively choose *Deploy from a branch*, branch `main`, folder `/project-page`.

## Rebuilding the video

The video is generated from real 2D runs of the released code, not from a
hand-drawn illustration: the particles follow the ODE of trained velocity fields,
and the MMD curves come from those runs.

```bash
# from the repository root, with the pfm package importable
export PYTHONPATH=$PWD

# 1. train the 2D models (CPU is fine; ~7 min with 3 seeds)
python project-page/build/make_assets.py --steps 4000 --gmm-steps 6000 --seeds 3

# 2. render 90 s at 1920x1080 / 30 fps (~3 min)
python project-page/build/make_video.py --out project-page/assets/video/pfm_concept.mp4

# 2b. web encodes (poster + VP9 fallback for browsers without H.264)
ffmpeg -y -ss 17.5 -i project-page/assets/video/pfm_concept.mp4 -frames:v 1 -q:v 3 \
       project-page/assets/video/pfm_concept_poster.jpg
ffmpeg -y -i project-page/assets/video/pfm_concept.mp4 -c:v libvpx-vp9 -crf 34 -b:v 0 \
       -row-mt 1 -cpu-used 2 project-page/assets/video/pfm_concept.webm

# 3. refresh the web-sized figures from the paper sources
python project-page/build/make_web_assets.py
```

`make_video.py --preview 3,14,33 --audit` dumps single frames to
`build/preview/` instead of encoding, which is handy when tweaking the layout.

## Figures

Every figure in `assets/figures/` is taken from the paper (same files as the
arXiv source) and only resized for the web. Figure 1 is rendered from the
vector original at 2200 px.

## License

The page, the build scripts and the video are released under the same terms as
the repository (Apache 2.0). The paper figures and the text they illustrate are
CC BY 4.0.
