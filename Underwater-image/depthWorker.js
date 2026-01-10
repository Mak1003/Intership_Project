importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = false;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let session = null;

async function loadModel() {
  session = await ort.InferenceSession.create(
    "midas_small.onnx",
    { executionProviders: ["wasm"] }
  );
}

loadModel();

onmessage = async (e) => {
  if (!session) return;

  const { imageData, width, height } = e.data;

  const input = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    input[i * 3 + 0] = imageData[i * 4 + 0] / 255;
    input[i * 3 + 1] = imageData[i * 4 + 1] / 255;
    input[i * 3 + 2] = imageData[i * 4 + 2] / 255;
  }

  const tensor = new ort.Tensor("float32", input, [1, 3, height, width]);

  // ✅ FIX: use real input name
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;

  const output = await session.run(feeds);
  const depth = output[session.outputNames[0]].data;

  postMessage({ depth, width, height });
};
