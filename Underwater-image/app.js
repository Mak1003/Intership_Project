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
const canvas = document.getElementById("glCanvas");
const gl = canvas.getContext("webgl2");

if (!gl) {
  alert("WebGL2 not supported");
}
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

void main() {
  vec3 J = texture(u_image, v_uv).rgb;

  float vertical = v_uv.y;
  float edge = edgeStrength(v_uv);
  float center = 1.0 - distance(v_uv, vec2(0.5));

  float depth = 
      0.6 * vertical +
      0.25 * (1.0 - edge) +
      0.15 * center;

  depth = clamp(depth, 0.0, 1.0);

  vec3 t;
  t.r = exp(-u_attenuation.r * depth);
  t.g = exp(-u_attenuation.g * depth);
  t.b = exp(-u_attenuation.b * depth);

  vec3 B = u_waterColor * (1.0 - exp(-u_haze * depth));

  vec3 underwater = J * t + B;
  vec3 finalColor = mix(J, underwater, u_intensity);

  outColor = vec4(finalColor, 1.0);
}`;

function compile(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
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

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  -1,-1, 1,-1, -1,1,
  -1,1, 1,-1, 1,1
]), gl.STATIC_DRAW);

const posLoc = gl.getAttribLocation(program, "a_position");
gl.enableVertexAttribArray(posLoc);
gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

let texture = gl.createTexture();

function loadTexture(img) {
  canvas.width = img.width;
  canvas.height = img.height;
  gl.viewport(0, 0, canvas.width, canvas.height);

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGB,
    gl.RGB, gl.UNSIGNED_BYTE, img
  );

  render();
}

function render() {
  const preset = PRESETS[presetSelect.value];

  gl.uniform1f(gl.getUniformLocation(program, "u_intensity"), intensity.value);
  gl.uniform2f(gl.getUniformLocation(program, "u_resolution"), canvas.width, canvas.height);
  gl.uniform3fv(gl.getUniformLocation(program, "u_waterColor"), preset.waterColor);
  gl.uniform3fv(gl.getUniformLocation(program, "u_attenuation"), preset.attenuation);
  gl.uniform1f(gl.getUniformLocation(program, "u_haze"), preset.haze);

  gl.drawArrays(gl.TRIANGLES, 0, 6);
}
const imageInput = document.getElementById("imageInput");
const presetSelect = document.getElementById("preset");
const intensity = document.getElementById("intensity");

imageInput.onchange = e => {
  const img = new Image();
  img.onload = () => loadTexture(img);
  img.src = URL.createObjectURL(e.target.files[0]);
};

presetSelect.oninput = render;
intensity.oninput = render;
document.getElementById("download").onclick = () => {
  const a = document.createElement("a");
  a.download = "underwater.png";
  a.href = canvas.toDataURL();
  a.click();
};
