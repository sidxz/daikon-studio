/*
 * A WebGL2 point renderer for the chemical-space map: two layers (the protocol's
 * compounds, then a run's on top), a camera, and one sprite shape per style.
 * Ported from ChemCellar's cluster-map renderer.
 *
 * ponytail: hand-written, because one scatter plot does not justify a library.
 * If a second WebGL graphic is ever approved, move to a small one (ogl) rather
 * than growing this.
 */

import type { Rgba } from "./color";
import type { View } from "./view";

/** Shared by the shader and the hosts: the order of `colors` follows these ids. */
export const STYLE = { train: 0, validation: 1, test: 2, runIn: 3, runOut: 4 } as const;
export const STYLE_COUNT = 5;
/** Sprite diameter per style, in CSS pixels. Dense training clouds stay small. */
const SIZES = [2.5, 2.5, 3, 9, 10];

const VERTEX = `#version 300 es
in vec2 a_pos;
in float a_style;
uniform vec2 u_center;
uniform float u_scale;
uniform vec2 u_resolution;
uniform float u_dpr;
uniform float u_progress;
uniform float u_pointScale;
uniform float u_sizes[${STYLE_COUNT}];
uniform vec4 u_colors[${STYLE_COUNT}];
out vec4 v_color;
flat out int v_style;
void main() {
  int style = int(a_style + 0.5);
  vec2 screen = vec2(
    (a_pos.x - u_center.x) * u_scale + u_resolution.x * 0.5,
    u_resolution.y * 0.5 - (a_pos.y - u_center.y) * u_scale
  );
  vec2 clip = screen / u_resolution * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  bool run = style >= 3;
  float grow = 1.0;
  if (run) {
    float t = clamp((u_progress - 0.35) / 0.65, 0.0, 1.0);
    grow = 1.0 - pow(1.0 - t, 3.0);
  }
  gl_PointSize = max(1.0, u_sizes[style] * (run ? 1.0 : u_pointScale) * u_dpr * grow);
  v_color = u_colors[style];
  v_color.a *= run ? step(0.001, grow) : clamp(u_progress / 0.5, 0.0, 1.0);
  v_style = style;
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
in vec4 v_color;
flat in int v_style;
out vec4 color;
void main() {
  vec2 p = gl_PointCoord - 0.5;
  float r = length(p);
  float aa = max(fwidth(r), 0.02);
  float a = 1.0 - smoothstep(0.5 - aa, 0.5, r);
  if (v_style == 4) {
    // Outside the applicability domain: a dashed ring, the mark's own "predicted" stroke.
    a *= smoothstep(0.30 - aa, 0.30, r) * step(0.0, sin(atan(p.y, p.x) * 5.0));
  }
  a *= v_color.a;
  if (a <= 0.0) discard;
  color = vec4(v_color.rgb * a, a);
}`;

export interface MapRenderer {
  /** Layer 0 draws first (the protocol's compounds), layer 1 on top (a run's). */
  setLayer(index: 0 | 1, positions: Float32Array, styles: Float32Array): void;
  setColors(colors: Rgba[]): void;
  /** Multiplier for the dataset layer's dots: larger when there are few. */
  setPointScale(scale: number): void;
  /** Match the drawing buffer to the canvas. Returns the size in CSS pixels. */
  resize(): [width: number, height: number];
  /** `progress` 0..1 is the entrance; 1 is at rest. */
  draw(view: View, progress: number): void;
  dispose(): void;
}

/** Null when WebGL2 is unavailable or the shaders fail, so the host shows its fallback. */
export function createMapRenderer(canvas: HTMLCanvasElement): MapRenderer | null {
  let gl: WebGL2RenderingContext | null = null;
  try {
    gl = canvas.getContext("webgl2", { antialias: true, alpha: true, premultipliedAlpha: true });
  } catch {
    return null;
  }
  if (!gl) return null;
  const context = gl;

  const compile = (type: number, source: string) => {
    const shader = context.createShader(type);
    if (!shader) throw new Error("could not create a shader");
    context.shaderSource(shader, source);
    context.compileShader(shader);
    if (!context.getShaderParameter(shader, context.COMPILE_STATUS)) {
      throw new Error(context.getShaderInfoLog(shader) ?? "shader did not compile");
    }
    return shader;
  };

  let program: WebGLProgram;
  try {
    program = context.createProgram();
    context.attachShader(program, compile(context.VERTEX_SHADER, VERTEX));
    context.attachShader(program, compile(context.FRAGMENT_SHADER, FRAGMENT));
    context.linkProgram(program);
    if (!context.getProgramParameter(program, context.LINK_STATUS)) {
      throw new Error(context.getProgramInfoLog(program) ?? "program did not link");
    }
  } catch {
    return null;
  }
  context.useProgram(program);

  const positionLocation = context.getAttribLocation(program, "a_pos");
  const styleLocation = context.getAttribLocation(program, "a_style");
  const layers = [0, 1].map(() => {
    const vao = context.createVertexArray();
    const positions = context.createBuffer();
    const styles = context.createBuffer();
    context.bindVertexArray(vao);
    context.bindBuffer(context.ARRAY_BUFFER, positions);
    context.enableVertexAttribArray(positionLocation);
    context.vertexAttribPointer(positionLocation, 2, context.FLOAT, false, 0, 0);
    context.bindBuffer(context.ARRAY_BUFFER, styles);
    context.enableVertexAttribArray(styleLocation);
    context.vertexAttribPointer(styleLocation, 1, context.FLOAT, false, 0, 0);
    context.bindVertexArray(null);
    return { vao, positions, styles, count: 0 };
  });

  const uniform = {
    center: context.getUniformLocation(program, "u_center"),
    scale: context.getUniformLocation(program, "u_scale"),
    resolution: context.getUniformLocation(program, "u_resolution"),
    dpr: context.getUniformLocation(program, "u_dpr"),
    progress: context.getUniformLocation(program, "u_progress"),
    pointScale: context.getUniformLocation(program, "u_pointScale"),
    sizes: context.getUniformLocation(program, "u_sizes"),
    colors: context.getUniformLocation(program, "u_colors"),
  };
  context.uniform1fv(uniform.sizes, SIZES);
  context.uniform1f(uniform.pointScale, 1);
  context.enable(context.BLEND);
  context.blendFunc(context.ONE, context.ONE_MINUS_SRC_ALPHA);
  context.clearColor(0, 0, 0, 0);

  let width = 0;
  let height = 0;
  let dpr = 1;

  return {
    setLayer(index, positions, styles) {
      const layer = layers[index];
      context.bindBuffer(context.ARRAY_BUFFER, layer.positions);
      context.bufferData(context.ARRAY_BUFFER, positions, context.STATIC_DRAW);
      context.bindBuffer(context.ARRAY_BUFFER, layer.styles);
      context.bufferData(context.ARRAY_BUFFER, styles, context.STATIC_DRAW);
      layer.count = styles.length;
    },
    setColors(colors) {
      context.useProgram(program);
      context.uniform4fv(uniform.colors, new Float32Array(colors.slice(0, STYLE_COUNT).flat()));
    },
    setPointScale(scale) {
      context.useProgram(program);
      context.uniform1f(uniform.pointScale, scale);
    },
    resize() {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      context.viewport(0, 0, canvas.width, canvas.height);
      return [width, height];
    },
    draw(view, progress) {
      context.useProgram(program);
      context.uniform2f(uniform.center, view.cx, view.cy);
      context.uniform1f(uniform.scale, view.scale);
      context.uniform2f(uniform.resolution, width, height);
      context.uniform1f(uniform.dpr, dpr);
      context.uniform1f(uniform.progress, progress);
      context.clear(context.COLOR_BUFFER_BIT);
      for (const layer of layers) {
        if (layer.count === 0) continue;
        context.bindVertexArray(layer.vao);
        context.drawArrays(context.POINTS, 0, layer.count);
      }
      context.bindVertexArray(null);
    },
    dispose() {
      for (const layer of layers) {
        context.deleteBuffer(layer.positions);
        context.deleteBuffer(layer.styles);
        context.deleteVertexArray(layer.vao);
      }
      context.deleteProgram(program);
    },
  };
}
