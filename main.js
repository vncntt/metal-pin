// Back to our original working code (no imports)
let scene, camera, renderer;
let pins = [];
let mouse = new THREE.Vector2();
let video;
let isSceneReady = false;
let ortSession;
let frameCanvas, frameCtx, depthCanvas, depthCtx;
let isProcessing = false;  // Flag to prevent overlapping processing
let controls;

// Add a flag to track if we've processed the depth data
let depthProcessed = false;

// Add a new flag to track if we've received depth data
let depthDataReceived = false;

// Add this flag at the top with other globals
let firstUpdateAttempt = true;

// Add a simple test to see if Transformers loaded
console.log("Transformers available:", typeof pipeline !== 'undefined');

// Grid settings
const gridSize = 40;  // number of pins in each row/column
const spacing = 0.8;  // space between pins
const pinRadius = 0.1;
const pinHeight = 1.0;

// Add these functions near the top of your file
const preprocess = (input_imageData, width, height) => {
    var floatArr = new Float32Array(width * height * 3);
    var floatArr1 = new Float32Array(width * height * 3);
    var floatArr2 = new Float32Array(width * height * 3);

    var j = 0;
    for (let i = 1; i < input_imageData.data.length + 1; i++) {
        if (i % 4 != 0) {
            floatArr[j] = input_imageData.data[i - 1] / 255; // red color
            j = j + 1;
        }
    }

    var k = 0;
    for (let i = 0; i < floatArr.length; i += 3) {
        floatArr2[k] = floatArr[i]; // red color
        k = k + 1;
    }
    var l = k;
    for (let i = 1; i < floatArr.length; i += 3) {
        floatArr2[l] = floatArr[i]; // green color
        l = l + 1;
    }
    var m = l;
    for (let i = 2; i < floatArr.length; i += 3) {
        floatArr2[m] = floatArr[i]; // blue color
        m = m + 1;
    }
    return floatArr2;
};

const postprocess = (tensor) => {
    console.log("Tensor dims:", tensor.dims);
    console.log("Tensor data type:", tensor.data.constructor.name);
    
    const channels = tensor.dims[1];
    const height = tensor.dims[2];
    const width = tensor.dims[3];
    
    console.log("Processing dimensions:", { channels, height, width });

    const imageData = new ImageData(width, height);
    const data = imageData.data;

    const tensorData = new Float32Array(tensor.data.buffer);
    console.log("Raw tensor data sample:", tensorData.slice(0, 10));
    
    let max_depth = 0;
    let min_depth = Infinity;

    // Find the min and max depth values
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const tensorIndex = (0 * height + h) * width + w;
            const value = tensorData[tensorIndex];
            if (value > max_depth) max_depth = value;
            if (value < min_depth) min_depth = value;
        }
    }
    
    console.log("Depth range:", { min_depth, max_depth });

    // Normalize and fill ImageData
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const tensorIndex = (0 * height + h) * width + w;
            const value = tensorData[tensorIndex];
            const depth = ((value - min_depth) / (max_depth - min_depth)) * 255;

            const pixelIndex = (h * width + w) * 4;
            data[pixelIndex] = Math.round(depth);     // R
            data[pixelIndex + 1] = Math.round(depth); // G
            data[pixelIndex + 2] = Math.round(depth); // B
            data[pixelIndex + 3] = 255;               // A
        }
    }
    
    console.log("Final ImageData dimensions:", imageData.width, "x", imageData.height);
    console.log("Sample of final pixel values:", data.slice(0, 20));

    return imageData;
};

async function setupWebcam() {
    console.log("Setting up webcam...");
    video = document.createElement('video');
    video.style.display = 'block';  // Make video visible
    document.getElementById('webcam-view').appendChild(video);

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        console.log("Webcam started successfully");
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

async function testDepthEstimation() {
    console.log("Starting depth estimation test...");
    const startTime = performance.now();
    
    try {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            // Draw webcam frame
            const frameStart = performance.now();
            frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
            const imageData = frameCtx.getImageData(0, 0, frameCanvas.width, frameCanvas.height);
            console.log("Frame capture time:", performance.now() - frameStart, "ms");
            
            // Process through ONNX
            const preprocessStart = performance.now();
            const preprocessedData = preprocess(imageData, frameCanvas.width, frameCanvas.height);
            console.log("Preprocessing time:", performance.now() - preprocessStart, "ms");
            
            const inferenceStart = performance.now();
            const input = new ort.Tensor(
                new Float32Array(preprocessedData), 
                [1, 3, frameCanvas.width, frameCanvas.height]
            );
            const result = await ortSession.run({ image: input });
            console.log("ONNX inference time:", performance.now() - inferenceStart, "ms");
            
            // Postprocess and display
            const postprocessStart = performance.now();
            const depthImageData = postprocess(result.depth);
            depthCtx.putImageData(depthImageData, 0, 0);
            console.log("Postprocessing time:", performance.now() - postprocessStart, "ms");
            
            // After putting depth data
            console.log("Depth data written to canvas");
            depthDataReceived = true;  // Set flag when we have depth data
            
            console.log("Total processing time:", performance.now() - startTime, "ms");
            return true;
        }
    } catch (error) {
        console.error("Test failed:", error);
        console.error("Error details:", error.stack);
        return false;
    }
}

function setupCanvases() {
    // Setup frame canvas
    frameCanvas = document.createElement('canvas');
    frameCtx = frameCanvas.getContext('2d');
    frameCanvas.width = 256;
    frameCanvas.height = 256;
    document.getElementById('frame-view').appendChild(frameCanvas);
    
    // Setup depth canvas
    depthCanvas = document.createElement('canvas');
    depthCtx = depthCanvas.getContext('2d');
    depthCanvas.width = 256;
    depthCanvas.height = 256;
    document.getElementById('depth-view').appendChild(depthCanvas);
}

async function init() {
    console.log("Testing ONNX availability:", typeof ort !== 'undefined');
    
    try {
        ortSession = await ort.InferenceSession.create(
            "https://cdn.glitch.me/0f5359e2-6022-421b-88f7-13e276d0fb33/depthanything-quant.onnx"
        );
        console.log("ONNX model loaded successfully");
    } catch (error) {
        console.error("Error loading ONNX model:", error);
    }

    console.log("Initializing...");
    await setupWebcam();
    setupCanvases();
    
    // Just run one test after webcam is ready
    setTimeout(async () => {
        console.log("Starting single frame capture...");
        const testResult = await testDepthEstimation();
        console.log("Depth estimation complete, result:", testResult);
        
        // Stop the webcam after capture
        const stream = video.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        console.log("Webcam stopped");
        
    }, 2000);
    
    // Setup scene
    scene = new THREE.Scene();
    console.log("Scene created");
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    console.log("Camera created");
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth/2, window.innerHeight/2);  // Half size
    document.getElementById('pins-view').appendChild(renderer.domElement);

    // Add OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; // Add smooth damping
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    controls.maxPolarAngle = Math.PI / 2;
    console.log("Orbit controls added");

    // Add lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(0, 5, 5);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x404040));

    // Create pin grid
    const geometry = new THREE.CylinderGeometry(pinRadius, pinRadius, pinHeight, 8);
    const material = new THREE.MeshPhongMaterial({ color: 0x888888 });

    // Calculate grid offset to center it
    const offset = (gridSize * spacing) / 2;

    for(let x = 0; x < gridSize; x++) {
        for(let y = 0; y < gridSize; y++) {
            const pin = new THREE.Mesh(geometry, material);
            pin.position.x = (x * spacing) - offset;
            pin.position.z = (y * spacing) - offset;
            pin.position.y = pinHeight / 2;  // Start at half height
            
            // Store the original height for reference
            pin.userData.originalY = pin.position.y;
            pins.push(pin);
            scene.add(pin);
        }
    }

    // Position camera
    camera.position.x = 15;
    camera.position.y = 15;
    camera.position.z = 15;
    camera.lookAt(0, 0, 0);

    isSceneReady = true;  // Set flag when everything is ready
    console.log("Initialization complete");
}

function updatePins() {
    // Only log on first attempt
    if (firstUpdateAttempt) {
        console.log("UpdatePins called. Depth data received:", depthDataReceived, 
                    "DepthCtx exists:", !!depthCtx, 
                    "Already processed:", depthProcessed);
        firstUpdateAttempt = false;
    }
                
    if (!depthCtx || depthProcessed || !depthDataReceived) {
        return;
    }
    
    console.log("Starting pin updates...");
    const depthData = depthCtx.getImageData(0, 0, depthCanvas.width, depthCanvas.height).data;
    
    // Verify we have valid depth data
    const nonZeroDepth = depthData.some(val => val > 0);
    console.log("Depth data present:", nonZeroDepth);
    console.log("Depth data length:", depthData.length);
    console.log("First 20 depth values:", Array.from(depthData.slice(0, 20)));
    
    // Calculate grid cell size in pixels
    const skipX = depthCanvas.width / gridSize;
    const skipY = depthCanvas.height / gridSize;
    console.log("Pixels per grid cell:", { skipX, skipY });
    
    // Debug: draw sampling points on depth canvas
    const debugCtx = depthCtx;
    debugCtx.strokeStyle = 'red';
    
    let maxDepthSeen = 0;
    let minDepthSeen = 1;
    
    pins.forEach((pin, index) => {
        const gridX = index % gridSize;
        const gridY = Math.floor(index / gridSize);
        
        // Calculate centered sampling point
        const x = Math.floor((gridX + 0.5) * skipX);
        const y = Math.floor((gridY + 0.5) * skipY);
        
        // Clamp to image bounds
        const safeX = Math.min(Math.max(x, 0), depthCanvas.width - 1);
        const safeY = Math.min(Math.max(y, 0), depthCanvas.height - 1);
        
        const pixelIndex = (safeY * depthCanvas.width + safeX) * 4;
        const depth = depthData[pixelIndex] / 255.0;
        
        // Track min/max depth
        maxDepthSeen = Math.max(maxDepthSeen, depth);
        minDepthSeen = Math.min(minDepthSeen, depth);
        
        // Debug: draw sampling point
        debugCtx.beginPath();
        debugCtx.arc(safeX, safeY, 1, 0, Math.PI * 2);
        debugCtx.stroke();
        
        // Scale pin (CHANGED SCALE FROM 2.0 TO 5.0)
        const heightScale = 1.0 + (depth * 5.0);
        pin.scale.y = heightScale;
        pin.position.y = pin.userData.originalY * heightScale;
        
        // Log some sample pins
        if (index === 0 || index === gridSize-1 || index === gridSize*gridSize-1) {
            console.log(`Pin ${index}:
                grid(${gridX},${gridY})
                pixel(${safeX},${safeY})
                depth=${depth.toFixed(3)}
                scale=${heightScale.toFixed(3)}
                finalY=${pin.position.y.toFixed(3)}`);
        }
    });
    
    console.log(`Depth range seen: ${minDepthSeen.toFixed(3)} to ${maxDepthSeen.toFixed(3)}`);
    
    // Set flag to prevent further updates
    depthProcessed = true;
    console.log("Pin update complete. First pin height:", pins[0].position.y);
    console.log("Last pin height:", pins[pins.length-1].position.y);
}

function animate() {
    requestAnimationFrame(animate);
    if (!isSceneReady) {
        return;
    }
    
    // Only log if we're about to do the update
    if (!depthProcessed && depthDataReceived) {
        console.log("Attempting to update pins...");
    }
    
    updatePins();
    controls.update();
    renderer.render(scene, camera);
}


console.log("About to call init");
init().then(() => {
    console.log("Starting animation loop");
    animate();
}).catch(error => {
    console.error("Error during initialization:", error);
});