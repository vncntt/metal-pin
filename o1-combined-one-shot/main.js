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
  const gridSize = 80;
  const spacing = 0.3;
  const pinRadius = 0.1;
  const pinHeight = 1.0;
  
  // For reading frames
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const outputContext = outputCanvas.getContext("2d", { willReadFrequently: true });
  
  let model;
  let processor;
  let isProcessing = false;
  let previousTime;
  let depthData;   // We'll store the raw depth float array here
  let depthWidth;  // The model’s depth-map width
  let depthHeight; // The model’s depth-map height
  
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
  
  // 4) Three.js scene for pins
  function initPinsScene() {
    scene = new THREE.Scene();
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
    const geometry = new THREE.CylinderGeometry(pinRadius, pinRadius, pinHeight, 8);
    const material = new THREE.MeshPhongMaterial({ color: 0x888888 });
    const offset = (gridSize * spacing) / 2;
  
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        const pin = new THREE.Mesh(geometry, material);
        pin.position.x = x * spacing - offset;
        pin.position.z = y * spacing - offset;
        pin.position.y = pinHeight / 2;
        pin.userData.originalY = pin.position.y;
        pins.push(pin);
        scene.add(pin);
      }
    }
  
    // Camera location
    camera.position.set(15, 15, 20);
    camera.lookAt(0, 0, 0);
  }
  
  // 5) Update pins from new depth
  function updatePins() {
    if (!depthData || !pins.length) return;
  
    // We know min/max from the normalization step below; let’s find them ourselves
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < depthData.length; i++) {
      const v = depthData[i];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    const range = maxVal - minVal;
  
    // Each pin samples from the scaled pixel location
    const skipX = depthWidth / gridSize;
    const skipY = depthHeight / gridSize;
  
    for (let i = 0; i < pins.length; i++) {
      const pin = pins[i];
      const gx = i % gridSize; 
      const gy = Math.floor(i / gridSize);
  
      const x = Math.floor((gx + 0.5) * skipX);
      const y = Math.floor((gy + 0.5) * skipY);
  
      // Clamp for safety
      const safeX = Math.min(Math.max(0, x), depthWidth - 1);
      const safeY = Math.min(Math.max(0, y), depthHeight - 1);
      const idx = safeY * depthWidth + safeX;
      
      // Normalize to [0..1]
      const rawD = depthData[idx];
      const norm = (rawD - minVal) / range;
  
      // Scale pin
      const heightScale = 1.0 + norm * 5.0; 
      pin.scale.y = heightScale;
      pin.position.y = pin.userData.originalY * heightScale;
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
      // Just a simple grayscale shading
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
  