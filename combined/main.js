import {
  AutoModel,
  AutoImageProcessor,
  RawImage,
} from "@huggingface/transformers";

// Check if we can use GPU fp16
async function hasFp16() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter.features.has("shader-f16");
  } catch {
    return false;
  }
}

// Elements from the DOM
const statusEl = document.getElementById("status");
const canvas = document.createElement("canvas");
const outputCanvas = document.getElementById("output-canvas");
const video = document.getElementById("video");
const sizeSlider = document.getElementById("size");
const sizeLabel = document.getElementById("size-value");

// Setup for 3D pins
let scene, camera, renderer, controls;
let pins = [];

// === CHANGED: Update dimensions for more realism ===
const gridSize = 120;
const spacing = 0.4;
const pinRadius = 0.14;
const pinHeight = 5.0;
// Add border size for the frame
const borderSize = 2.0; // Extra space on each side
const basePlateSize = gridSize * spacing + borderSize * 2;

// For reading frames
const context = canvas.getContext("2d", { willReadFrequently: true });
const outputContext = outputCanvas.getContext("2d", { willReadFrequently: true });

let model;
let processor;
let isProcessing = false;
let previousTime;
let depthData;   // We'll store the raw depth float array here
let depthWidth;  // The model's depth-map width
let depthHeight; // The model's depth-map height

// 1) Load the model
statusEl.textContent = "Loading model...";
const model_id = "onnx-community/depth-anything-v2-small";

try {
  model = await AutoModel.from_pretrained(model_id, {
    device: "webgpu",
    dtype: (await hasFp16()) ? "fp16" : "fp32",
  });
} catch (err) {
  statusEl.textContent = err.message;
  alert(err.message);
  throw err;
}

processor = await AutoImageProcessor.from_pretrained(model_id);
statusEl.textContent = "Ready";

// 2) Allow user to change input resolution
let size = 504;
processor.size = { width: size, height: size };
sizeSlider.addEventListener("input", () => {
  size = Number(sizeSlider.value);
  processor.size = { width: size, height: size };
  sizeLabel.textContent = size;
});
sizeSlider.disabled = false;

// 3) Setup video
function setStreamSize(w, h) {
  video.width = outputCanvas.width = canvas.width = w;
  video.height = outputCanvas.height = canvas.height = h;
}

// === Quadratic interpolation helper functions ===
function lagrangeInterpolate(y0, y1, y2, t) {
  // 1D Quadratic Lagrange interpolation for x=0,1,2 => find value at x=t in [0..2]
  // L(t) = Σ [y_i * l_i(t)], i = 0..2
  // where l_0(t) = (t-1)(t-2) / ((0-1)*(0-2)), etc.
  const c0 = y0 * ((t - 1) * (t - 2)) / ((0 - 1) * (0 - 2));
  const c1 = y1 * ((t - 0) * (t - 2)) / ((1 - 0) * (1 - 2));
  const c2 = y2 * ((t - 0) * (t - 1)) / ((2 - 0) * (2 - 1));
  return c0 + c1 + c2;
}

/**
 * Quadratic interpolation in 2D over a 3×3 neighborhood.
 * (u, v) can be fractional. We clamp so we don't go out of bounds.
 */
function sampleDepthQuadratic(u, v, data, w, h) {
  // Make sure we stay within [1, w-2] and [1, h-2] so we can safely grab neighbors
  u = Math.max(1, Math.min(w - 2, u));
  v = Math.max(1, Math.min(h - 2, v));

  // x0,y0 is the top-left corner in the 3×3 block
  const x0 = Math.floor(u) - 1;
  const y0 = Math.floor(v) - 1;
  const x1 = x0 + 1;
  const x2 = x0 + 2;
  const y1 = y0 + 1;
  const y2 = y0 + 2;

  // Fractional offsets in [0..2]
  const fx = u - (x0);
  const fy = v - (y0);

  // Safe accessor
  function d(x, y) {
    return data[y * w + x];
  }

  // 1D interpolation along x for each of the three rows
  const row0 = lagrangeInterpolate(d(x0, y0), d(x1, y0), d(x2, y0), fx);
  const row1 = lagrangeInterpolate(d(x0, y1), d(x1, y1), d(x2, y1), fx);
  const row2 = lagrangeInterpolate(d(x0, y2), d(x1, y2), d(x2, y2), fx);

  // Then 1D interpolation in y for those three results
  return lagrangeInterpolate(row0, row1, row2, fy);
}

// 4) Three.js scene for pins
function initPinsScene() {
  scene = new THREE.Scene();

  // === CHANGED: Make the background white ===
  scene.background = new THREE.Color(0xffffff);

  camera = new THREE.PerspectiveCamera(75, 800 / 600, 0.1, 1000);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(800, 600);

  const threeContainer = document.getElementById("three-container");
  threeContainer.appendChild(renderer.domElement);

  // Orbit controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = 5;
  controls.maxDistance = 50;
  controls.maxPolarAngle = Math.PI / 2;

  // Lights
  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(0, 5, 5);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0x404040));

  // Build pin grid
  const pinBodyRadius = pinRadius;
  const pinTipRadius = pinRadius * 1.4;
  const pinBodyGeometry = new THREE.CylinderGeometry(pinBodyRadius, pinBodyRadius, pinHeight - pinTipRadius, 8);
  const pinTipGeometry = new THREE.SphereGeometry(pinTipRadius, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2);
  const material = new THREE.MeshPhongMaterial({ 
    color: 0xCCCCCC,      // Lighter silver color
    shininess: 50,       // Higher shininess for more specular highlights
    specular: 0x666666,   // Add specular highlights
    metalness: 0.6,       // Make it look more metallic
  });

  // Shift body geometry so bottom is at y=0
  pinBodyGeometry.translate(0, (pinHeight - pinTipRadius) / 2, 0);
  // Shift tip geometry to top of body
  pinTipGeometry.translate(0, pinHeight - pinTipRadius, 0);

  const offset = (gridSize * spacing) / 2;

  for (let x = 0; x < gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      // Create a group to hold both body and tip
      const pinGroup = new THREE.Group();
      
      const pinBody = new THREE.Mesh(pinBodyGeometry, material);
      const pinTip = new THREE.Mesh(pinTipGeometry, material);
      
      pinGroup.add(pinBody);
      pinGroup.add(pinTip);
      
      pinGroup.position.x = x * spacing - offset;
      pinGroup.position.z = y * spacing - offset;
      // Start pins at y=0
      pinGroup.position.y = 0;
      
      pins.push(pinGroup);
      scene.add(pinGroup);
    }
  }

  // === CHANGED: Make base plate larger with border ===
  const baseThickness = 0.2;
  const baseGeometry = new THREE.BoxGeometry(basePlateSize, baseThickness, basePlateSize);
  const baseMaterial = new THREE.MeshPhongMaterial({ color: 0x000000 });
  const base = new THREE.Mesh(baseGeometry, baseMaterial);
  base.position.set(0, baseThickness / 2 - 0.01, 0);
  scene.add(base);

  // Add the support pillars
  const pillarHeight = pinHeight * 2 + 0.5;
  const pillarRadius = 0.8;
  const pillarCapRadius = pillarRadius * 1.2; // Slightly larger cap for aesthetic
  const pillarGeometry = new THREE.CylinderGeometry(pillarRadius, pillarRadius, pillarHeight - pillarCapRadius, 16);
  const pillarCapGeometry = new THREE.SphereGeometry(pillarCapRadius, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
  const pillarMaterial = new THREE.MeshPhongMaterial({ color: 0x333333 });

  // Position for the 6 pillars
  const pillarPositions = [
    [-basePlateSize/2 + borderSize/2, -basePlateSize/2 + borderSize/2],   // Front left
    [-basePlateSize/2 + borderSize/2, basePlateSize/2 - borderSize/2],    // Back left
    [basePlateSize/2 - borderSize/2, -basePlateSize/2 + borderSize/2],    // Front right
    [basePlateSize/2 - borderSize/2, basePlateSize/2 - borderSize/2],     // Back right
  ];

  // Add the glass pane first
  const glassThickness = 0.2;
  const glassGeometry = new THREE.BoxGeometry(basePlateSize, glassThickness, basePlateSize);
  const glassMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.5,     // CHANGED: Higher opacity
    roughness: 0.1,   // CHANGED: Slight roughness for more realistic look
    metalness: 0.1,   // CHANGED: Slight metalness for better reflections
    transmission: 0.7, // CHANGED: Less transmission to make it more visible
    thickness: glassThickness,
    clearcoat: 1.0,
    clearcoatRoughness: 0.1, // ADDED: Slight roughness to the clearcoat
    ior: 1.5,        // ADDED: Index of refraction for glass
  });

  const glass = new THREE.Mesh(glassGeometry, glassMaterial);
  glass.position.set(0, pillarHeight - pillarCapRadius, 0); // Position just below where caps will go
  scene.add(glass);

  // Then add pillars and caps
  pillarPositions.forEach(([x, z]) => {
    // Create the main pillar body
    const pillar = new THREE.Mesh(pillarGeometry, pillarMaterial);
    pillar.position.set(x, (pillarHeight - pillarCapRadius)/2, z);
    scene.add(pillar);

    // Add the rounded cap on top of the glass
    const cap = new THREE.Mesh(pillarCapGeometry, pillarMaterial);
    cap.position.set(x, pillarHeight - pillarCapRadius, z);
    scene.add(cap);
  });

  // Camera location
  camera.position.set(15, 15, 20);
  camera.lookAt(0, 0, 0);
}

// 5) Update pins from new depth
function updatePins() {
  if (!depthData || !pins.length) return;

  // Find min/max to normalize from raw depth
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < depthData.length; i++) {
    const v = depthData[i];
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  const range = maxVal - minVal;

  const skipX = depthWidth / (gridSize - 1);
  const skipY = depthHeight / (gridSize - 1);

  for (let i = 0; i < pins.length; i++) {
    const pinGroup = pins[i];
    const gx = i % gridSize;
    const gy = Math.floor(i / gridSize);

    const x = (gx + 0.5) * skipX;
    const y = (gy + 0.5) * skipY;

    const rawD = sampleDepthQuadratic(x, y, depthData, depthWidth, depthHeight);
    const norm = (rawD - minVal) / range;

    // Move the entire pin group
    pinGroup.position.y = 5.0 * norm;
  }
}

// 6) Animation loop for the 3D scene
function animate() {
  requestAnimationFrame(animate);
  updatePins();
  controls.update();
  renderer.render(scene, camera);
}

// 7) Process each video frame and run inference
async function runInference() {
  if (isProcessing) return;
  isProcessing = true;

  const { width, height } = canvas;
  context.drawImage(video, 0, 0, width, height);
  const currentFrame = context.getImageData(0, 0, width, height);
  const image = new RawImage(currentFrame.data, width, height, 4);

  // Pre-process
  const inputs = await processor(image);

  // Predict
  const { predicted_depth } = await model(inputs);
  depthData = predicted_depth.data; // Float32Array
  const [bs, oh, ow] = predicted_depth.dims;
  depthWidth = ow;
  depthHeight = oh;

  // Also show the depth map in outputCanvas (as alpha overlay)
  outputCanvas.width = ow;
  outputCanvas.height = oh;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < depthData.length; ++i) {
    const v = depthData[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  const imageData = new Uint8ClampedArray(4 * depthData.length);
  for (let i = 0; i < depthData.length; ++i) {
    const offset = 4 * i;
    // Simple grayscale shading
    const norm = (depthData[i] - min) / range;
    const gray = 255 * (1 - norm);
    imageData[offset + 0] = gray;
    imageData[offset + 1] = gray;
    imageData[offset + 2] = gray;
    imageData[offset + 3] = 255;
  }
  const outPixelData = new ImageData(imageData, ow, oh);
  outputContext.putImageData(outPixelData, 0, 0);

  // Show FPS
  if (previousTime !== undefined) {
    const fps = 1000 / (performance.now() - previousTime);
    statusEl.textContent = `FPS: ${fps.toFixed(2)}`;
  }
  previousTime = performance.now();

  isProcessing = false;
}

// 8) Continuously update
function loop() {
  window.requestAnimationFrame(loop);
  runInference();
}

// 9) Start up the video
navigator.mediaDevices
  .getUserMedia({ video: { width: 720, height: 720 } })
  .then((stream) => {
    video.srcObject = stream;
    video.play();

    // Use the real stream size or a scaled version
    const track = stream.getVideoTracks()[0];
    const { width, height } = track.getSettings();
    setStreamSize(width, height);

    // Initialize the Three.js pin scene
    initPinsScene();
    animate();

    // Start inference loop
    setTimeout(loop, 50);
  })
  .catch((error) => {
    alert(error);
  });