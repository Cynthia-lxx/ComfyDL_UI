/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Algorithm ported from osu.Game/Graphics/Backgrounds/TrianglesV2.cs:79-283 and
 * TriangleBorderData.cs:9-15 (UBO m_BorderData{Thickness, TexelSize}).
 * IMPORTANT: upstream GLSL text is NOT in this repository (it lives in the
 * ppy.osu.Framework / osu.Game.Resources NuGet packages). The shaders below are
 * independently written from the C# algorithm description; no shader source was copied.
 */

const VERTEX_SOURCE = `#version 300 es
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec3 aBary;
layout(location = 2) in vec4 aColour;
out vec3 vBary;
out vec4 vColour;
void main() {
  gl_Position = vec4(aPosition.x * 2.0 - 1.0, 1.0 - aPosition.y * 2.0, 0.0, 1.0);
  vBary = aBary;
  vColour = aColour;
}`;

const FRAGMENT_SOURCE = `#version 300 es
precision highp float;
in vec3 vBary;
in vec4 vColour;
layout(std140) uniform BorderData {
  float uThickness;
  vec2 uTexelSize;
};
out vec4 outColour;
void main() {
  float d = min(min(vBary.x, vBary.y), vBary.z);
  float aa = max(fwidth(d), uTexelSize.x) * 1.5;
  float border = 1.0 - smoothstep(uThickness, uThickness + aa, d);
  if (border <= 0.001) discard;
  outColour = vec4(vColour.rgb, vColour.a * border);
}`;

const BARYCENTRIC = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
const FLOATS_PER_VERTEX = 9; // 2 position + 3 barycentric + 4 colour
const MAX_TRIANGLES = Math.floor(65535 / 5);

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`[triangle-border] shader compile failed: ${info}`);
  }
  return shader;
}

function createProgram(gl) {
  const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SOURCE);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SOURCE);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`[triangle-border] link failed: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

/** Resolve a CSS custom property (possibly hsl()) into an RGBA tuple. */
function readColour(element, name) {
  const raw = getComputedStyle(element).getPropertyValue(name).trim();
  const probe = document.createElement('canvas').getContext('2d');
  probe.fillStyle = '#ffffff';
  probe.fillStyle = raw || '#ffffff';
  const value = probe.fillStyle;
  if (value.startsWith('#')) {
    const hex = value.slice(1);
    return [
      parseInt(hex.slice(0, 2), 16) / 255,
      parseInt(hex.slice(2, 4), 16) / 255,
      parseInt(hex.slice(4, 6), 16) / 255,
    ];
  }
  const parts = value.match(/[\d.]+/g);
  return parts && parts.length >= 3
    ? [Number(parts[0]) / 255, Number(parts[1]) / 255, Number(parts[2]) / 255]
    : [0.4, 0.8, 1];
}

function makeTriangle(random, randomY, size) {
  const x = random();
  const y = randomY ? random() * 1.2 - 0.1 : -0.1;
  return {
    x, y,
    size: size * (0.5 + random()),
    speed: 0.35 + random() * 0.8,
    shade: 0.45 + random() * 0.55,
  };
}

/**
 * Mount the triangle-border 视觉效果 on a canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {{density?:number, thickness?:number, velocity?:number, seed?:number}} [options]
 */
export function mountTriangles(canvas, options = {}) {
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
  if (!gl) {
    console.warn('[triangle-border] WebGL2 unavailable; 视觉效果 skipped');
    return { unmount() {}, setTheme() {}, setDensity() {}, available: false };
  }

  let program;
  try {
    program = createProgram(gl);
  } catch (error) {
    console.warn(error.message);
    return { unmount() {}, setTheme() {}, setDensity() {}, available: false };
  }

  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  const ubo = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

  const stride = FLOATS_PER_VERTEX * 4;
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 2 * 4);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 5 * 4);

  // std140: float at offset 0, vec2 must align to 8 -> one float of padding.
  const blockIndex = gl.getUniformBlockIndex(program, 'BorderData');
  if (blockIndex !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, blockIndex, 0);
  gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
  gl.bufferData(gl.UNIFORM_BUFFER, 16, gl.DYNAMIC_DRAW);
  gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ubo);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive, matching BlendingParameters.Additive

  let state = {
    density: options.density ?? 1,
    thickness: options.thickness ?? 0.02,
    velocity: options.velocity ?? 0.06,
    // triangleSize is in CSS pixels; buildVertices converts it to clip space.
    // (Previously a 0-1 value was divided by the canvas width, making every
    // triangle sub-pixel and effectively invisible.)
    triangleSize: options.triangleSize ?? 26,
  };
  let seed = options.seed ?? 20261113;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  let triangles = [];
  let colour = readColour(canvas, '--colour-1');
  let raf = 0;
  let last = performance.now();
  let disposed = false;

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
  };

  const targetCount = () => Math.max(16, Math.min(
    MAX_TRIANGLES,
    Math.floor(canvas.clientWidth * 0.12 * state.density),
  ));

  const replenish = (initial) => {
    const target = targetCount();
    while (triangles.length < target) triangles.push(makeTriangle(random, initial, state.triangleSize));
  };

  const buildVertices = () => {
    const width = canvas.clientWidth || 1;
    const height = canvas.clientHeight || 1;
    const data = new Float32Array(triangles.length * 3 * FLOATS_PER_VERTEX);
    let offset = 0;
    for (const t of triangles) {
      const w = t.size;
      const h = t.size * 0.866;
      const points = [
        [t.x, t.y],
        [t.x + w / width, t.y],
        [t.x + w / (2 * width), t.y + h / height],
      ];
      for (let i = 0; i < 3; i++) {
        data[offset++] = points[i][0];
        data[offset++] = points[i][1];
        data[offset++] = BARYCENTRIC[i][0];
        data[offset++] = BARYCENTRIC[i][1];
        data[offset++] = BARYCENTRIC[i][2];
        data[offset++] = colour[0] * t.shade;
        data[offset++] = colour[1] * t.shade;
        data[offset++] = colour[2] * t.shade;
        data[offset++] = 0.85 * t.shade;
      }
    }
    return data;
  };

  const frame = (now) => {
    if (disposed) return;
    resize();
    const elapsed = Math.min((now - last) / 1000, 0.05); // 大跳帧限补发
    last = now;

    for (let i = triangles.length - 1; i >= 0; i--) {
      const t = triangles[i];
      t.y += elapsed * state.velocity * t.speed;
      if (t.y > 1.1) triangles.splice(i, 1);
    }
    replenish(false);

    const vertices = buildVertices();
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);

    const texelSize = 1.5 / Math.max(1, canvas.height);
    gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, new Float32Array([state.thickness, 0, texelSize, texelSize]));
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ubo);

    gl.drawArrays(gl.TRIANGLES, 0, triangles.length * 3);
    raf = requestAnimationFrame(frame);
  };

  replenish(true);
  raf = requestAnimationFrame(frame);

  return {
    available: true,
    /** Local theme override: read theme colours from the canvas subtree. */
    setTheme(name) {
      if (name) canvas.dataset.theme = name;
      colour = readColour(canvas, '--colour-1');
    },
    setDensity(value) {
      state.density = Math.max(0, Number(value) || 0);
      const target = Math.min(triangles.length, targetCount());
      triangles.length = target;
    },
    setThickness(value) { state.thickness = Math.max(0, Number(value) || 0); },
    unmount() {
      disposed = true;
      cancelAnimationFrame(raf);
      gl.deleteBuffer(vbo);
      gl.deleteBuffer(ubo);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
    },
  };
}

export default mountTriangles;
