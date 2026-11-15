// Loads the MediaPipe models from local files (nothing is fetched from a CDN).
import { FilesetResolver, FaceLandmarker, ImageEmbedder } from '../vendor/mediapipe/vision_bundle.mjs';

const base = new URL('..', import.meta.url).href; // the site/ folder

let filesetPromise;
function fileset() {
  filesetPromise ??= FilesetResolver.forVisionTasks(base + 'vendor/mediapipe/wasm');
  return filesetPromise;
}

async function withGpuFallback(make) {
  try {
    return await make('GPU');
  } catch (e) {
    console.warn('GPU delegate failed, using CPU', e);
    return make('CPU');
  }
}

export async function createLandmarker(runningMode = 'IMAGE') {
  const fs = await fileset();
  return withGpuFallback((delegate) => FaceLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath: base + 'models/face_landmarker.task', delegate },
    runningMode,
    numFaces: 2,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  }));
}

export async function createEmbedder() {
  const fs = await fileset();
  return withGpuFallback((delegate) => ImageEmbedder.createFromOptions(fs, {
    baseOptions: { modelAssetPath: base + 'models/mobilenet_v3_large.tflite', delegate },
    runningMode: 'IMAGE',
    l2Normalize: true,
    quantize: false,
  }));
}
