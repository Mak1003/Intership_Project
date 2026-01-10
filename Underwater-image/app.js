/* =========================
   PRESETS
========================= */
const PRESETS = {
  OCEAN: {
    waterColor: [0.0, 0.35, 0.55],
    attenuation: [3.0, 1.5, 0.8],
    haze: 2.0
  },
  POOL: {
    waterColor: [0.2, 0.7, 0.8],
    attenuation: [2.0, 1.0, 0.5],
    haze: 1.2
  },
  DEEP: {
    waterColor: [0.0, 0.15, 0.3],
    attenuation: [4.0, 2.2, 1.0],
    haze: 3.0
  }
};

/* =========================
   WEBGL SETUP (FIRST!)
========================= */
const canvas = document.getElementById("glCanvas");
const gl = canvas.getContext("webgl2");

if (!gl) {
  alert("WebGL2 not supported");
  throw new Error("WebGL2 not supported");
}

/* =========================
   ML DEPTH WORKER
========================= */
const depthWorker = new Worker("depthWorker.js");

let depthTexture = gl.createTexture();
let useMLDepth = false;

depthWorker.onmessage = (e) => {
  const { depth, width, height } = e.data;

  // Normalize depth
  let min = Infinity, max = -Infinity;
  for (let v of depth) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }

  const normalized = new Uint8Array(depth.length);
  for (let i = 0; i < depth.length; i++) {
    const v = (depth[i] - min) / (max - min + 1e-6);
  normalized[i] = Math.max(0, Math.min(255, v * 255));
  }

  gl.bindTexture(gl.TEXTURE_2D, depthTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.R8,           // ✅ SAFE FORMAT
    width,
    height,
    0,
    gl.RED,
    gl.UNSIGNED_BYTE,
    normalized
  );

  useMLDepth = true;
  render();
};

function runMLDepth(img) {
  const size = 256;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");

  ctx.drawImage(img, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;

  depthWorker.postMessage({
    imageData: data,
    width: size,
    height: size
  });
}

/* =========================
   SHADERS
========================= */
const vertexSrc = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const fragmentSrc = `#version 300 es
precision highp float;

uniform sampler2D u_image;
uniform sampler2D u_depth;
uniform bool u_useMLDepth;

uniform vec2 u_resolution;
uniform float u_intensity;
uniform vec3 u_waterColor;
uniform vec3 u_attenuation;
uniform float u_haze;

in vec2 v_uv;
out vec4 outColor;

float luminance(vec3 c) {
  return dot(c, vec3(0.299, 0.587, 0.114));
}

float edgeStrength(vec2 uv) {
  vec2 px = 1.0 / u_resolution;
  float l = luminance(texture(u_image, uv).rgb);
  float r = luminance(texture(u_image, uv + vec2(px.x, 0.0)).rgb);
  float u = luminance(texture(u_image, uv + vec2(0.0, px.y)).rgb);
  return clamp(abs(l - r) + abs(l - u), 0.0, 1.0);
}

vec3 depthBlur(vec2 uv, float depth) {
  vec2 px = 1.0 / u_resolution;
  float r = depth * 2.0;
  vec3 s = vec3(0.0);
  s += texture(u_image, uv).rgb;
  s += texture(u_image, uv + px * vec2(r, 0.0)).rgb;
  s += texture(u_image, uv + px * vec2(-r, 0.0)).rgb;
  s += texture(u_image, uv + px * vec2(0.0, r)).rgb;
  s += texture(u_image, uv + px * vec2(0.0, -r)).rgb;
  return s / 5.0;
}

void main() {
  vec3 original = texture(u_image, v_uv).rgb;

  float depth;
  if (u_useMLDepth) {
    depth = texture(u_depth, v_uv).r;
  } else {
    float vertical = v_uv.y;
    float edge = edgeStrength(v_uv);
    float center = 1.0 - distance(v_uv, vec2(0.5));
    depth = 0.6 * vertical + 0.25 * (1.0 - edge) + 0.15 * center;
  }
  depth = clamp(depth, 0.0, 1.0);

  vec3 J = mix(original, depthBlur(v_uv, depth), depth * 0.6);

  vec3 t;
  t.r = exp(-u_attenuation.r * depth);
  t.g = exp(-u_attenuation.g * depth);
  t.b = exp(-u_attenuation.b * depth);

  vec3 B = u_waterColor * (1.0 - exp(-u_haze * depth));
  vec3 underwater = J * t + B;
  underwater = mix(underwater, u_waterColor, depth * 0.15);

  vec3 mixed = mix(original, underwater, u_intensity);
  vec3 finalColor = mix(vec3(0.5), mixed, 1.0 - depth * 0.25);
  finalColor = pow(finalColor, vec3(1.0 / 1.2));

  outColor = vec4(finalColor, 1.0);
}
`;

/* =========================
   PROGRAM
========================= */
function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
  }
  return s;
}

const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSrc));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSrc));
gl.linkProgram(program);
gl.useProgram(program);

/* =========================
   GEOMETRY
========================= */
const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  -1,-1,  1,-1, -1, 1,
  -1, 1,  1,-1,  1, 1
]), gl.STATIC_DRAW);

const posLoc = gl.getAttribLocation(program, "a_position");
gl.enableVertexAttribArray(posLoc);
gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

/* =========================
   IMAGE TEXTURE
========================= */
const imageTexture = gl.createTexture();

function loadTexture(img) {
  canvas.width = img.width;
  canvas.height = img.height;
  gl.viewport(0, 0, canvas.width, canvas.height);

  gl.bindTexture(gl.TEXTURE_2D, imageTexture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);

  render();
}

/* =========================
   RENDER
========================= */
function render() {
  const preset = PRESETS[presetSelect.value];

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, imageTexture);
  gl.uniform1i(gl.getUniformLocation(program, "u_image"), 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, depthTexture);
  gl.uniform1i(gl.getUniformLocation(program, "u_depth"), 1);

  gl.uniform1i(gl.getUniformLocation(program, "u_useMLDepth"), useMLDepth);

  gl.uniform1f(gl.getUniformLocation(program, "u_intensity"), intensity.value);
  gl.uniform2f(gl.getUniformLocation(program, "u_resolution"), canvas.width, canvas.height);
  gl.uniform3fv(gl.getUniformLocation(program, "u_waterColor"), preset.waterColor);
  gl.uniform3fv(gl.getUniformLocation(program, "u_attenuation"), preset.attenuation);
  gl.uniform1f(gl.getUniformLocation(program, "u_haze"), preset.haze);

  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

/* =========================
   UI
========================= */
const imageInput = document.getElementById("imageInput");
const presetSelect = document.getElementById("preset");
const intensity = document.getElementById("intensity");

imageInput.onchange = (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    loadTexture(img);
    runMLDepth(img);
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
};


presetSelect.oninput = render;
intensity.oninput = render;

/* =========================
   DOWNLOAD
========================= */
document.getElementById("download").onclick = () => {
  const a = document.createElement("a");
  a.download = "underwater.png";
  a.href = canvas.toDataURL();
  a.click();
};
