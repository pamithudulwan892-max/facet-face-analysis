# Facet - face analysis in the browser

Facet is a website that scans your face through your camera, estimates how a panel of raters would score the photo, and breaks the result down into symmetry, facial balance, skin clarity, the eye area and photo conditions. It then gives practical tips for the parts you can actually change.

Everything runs on your device. The camera feed is processed by models running in the browser, and no image or measurement is sent to a server.

## Features

- Live camera scan with on-screen guidance (distance, centring, head angle, light, holding still) that starts automatically when the face is lined up, and averages 8 frames
- Photo upload as an alternative, also processed locally
- Overall score out of 10 with an honest error range and a percentile against the dataset
- Breakdown of five factors, each marked as changeable or mostly fixed
- Tips ordered by how much they are likely to help for that scan
- Landmark map and a table of detailed measurements
- "How it works" page with accuracy, per-group accuracy, dataset details and limitations
- Works on mobile, and supports light and dark mode

## How it works

1. **Landmarks.** MediaPipe Face Landmarker finds 478 points on the face.
2. **Normalising.** The face is rotated so the eyes are level and scaled so the eye distance is 1.
3. **Features.** `site/js/features.js` computes 18 geometric features (ratios, angles, symmetry) and skin and image-quality statistics from an aligned 256x256 crop.
4. **Embedding.** MobileNet V3 (MediaPipe Image Embedder) turns the aligned crop into a 1280-value embedding.
5. **Model.** A ridge regression trained in Python combines the embedding, the landmark shape and the features into a predicted rating (1 to 5), shown on a 1 to 10 scale.

The same `features.js` file is used both to build the training data and on the live site, so the model always sees features calculated in the same way.

## Dataset

I used [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release), a public research dataset of 5,500 frontal face photos (Asian and Caucasian, male and female, aged 15 to 60), each rated 1 to 5 by 60 volunteers. It is released for non-commercial research use only, so this is a non-commercial student project. I chose a published dataset instead of scraping photos from the web, because scraped faces would be biometric data collected without consent.

> Liang, L., Lin, L., Jin, L., Xie, D. and Li, M. (2018). SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction. ICPR 2018.

The dataset itself is not included in this repository.

## Results

See `data/training_report.json` for the full numbers. On the official 60/40 test split, the combined model reaches a Pearson correlation of 0.85 with the average human rating and a mean absolute error of 0.28 on the 1 to 5 scale (2,200 test faces).

| Features | Inputs | Test correlation |
|---|---|---|
| Geometric ratios only | 18 | 0.70 |
| Skin measurements only | 5 | 0.39 |
| Landmark shape | 956 | 0.79 |
| Image embedding only | 1,280 | 0.80 |
| All combined (used) | 2,259 | 0.85 |

Accuracy is similar across the four groups in the test set (correlation 0.83 to 0.86).

## Limitations

- The dataset only includes Asian and Caucasian faces rated by a small group of raters, so results are less reliable for everyone else.
- Webcam noise, harsh light and beauty filters shift the skin and photo measurements.
- Attractiveness is subjective. The score is a model's opinion, not a measure of anyone's worth.

## Running it locally

The site is static, so any local web server works. The camera needs `localhost` or `https`.

```
python -m http.server 8080 --directory site
```

Then open http://localhost:8080.

## Retraining the model

1. Download SCUT-FBP5500 and extract it to `data/raw/SCUT-FBP5500_v2/`.
2. Run `python tools/extract_server.py` and open http://localhost:8765/tools/extract.html. The page runs MediaPipe on every image in the browser and saves the results to `data/landmarks/`.
3. Run `python tools/train.py`. It checks the landmark pairs, builds the geometry features with Node (`tools/build_geometry.mjs`), compares models, and writes `site/models/model.json`.

Requirements: Python with numpy, pandas and scikit-learn, and Node 22 or newer.

## Project structure

```
site/            the website (deploy this folder)
  js/features.js   shared feature extraction
  js/scoring.js    score, breakdown and tips
  js/scan.js       camera and photo flow
  models/          MediaPipe models and the trained model.json
  vendor/          MediaPipe Tasks Vision (local copy)
tools/           dataset extraction and training scripts
data/            training report and landmark pair list
```

## Built with

MediaPipe Tasks Vision, plain HTML, CSS and JavaScript, Python, NumPy, pandas and scikit-learn.
