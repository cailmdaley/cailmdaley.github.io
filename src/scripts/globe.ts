/**
 * The UNIONS mass-map globe: a celestial sphere textured with the
 * equirectangular sky baked by `scripts/globe.py`, drawn by one WebGL2 fragment
 * shader that ray-casts an orthographic sphere.
 *
 * It turns slowly on its own (not under prefers-reduced-motion), pauses while
 * off screen, and the visitor can drag to turn it and pinch or scroll to zoom.
 * Scrolling zooms only once the globe is selected (focused), or with a
 * trackpad pinch (ctrl+wheel), so it never takes over page scrolling.
 * Until the texture arrives, and without WebGL2, the baked still shows.
 *
 * Sky convention, as on the status page and in the demo video: seen from
 * outside, RA increases to the left.
 */

const VERT = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uSky;
uniform vec3 uC, uR, uU;
uniform vec2 uRes;
uniform float uZoom;
out vec4 outColor;
const float PI = 3.14159265358979;
void main() {
  vec2 p = (gl_FragCoord.xy / uRes * 2.0 - 1.0) / uZoom;
  float r = length(p);
  float px = 2.0 / (uRes.x * uZoom);
  float cover = 1.0 - smoothstep(1.0 - 1.5 * px, 1.0, r);
  if (cover <= 0.0) { outColor = vec4(0.0); return; }
  float mu = sqrt(max(0.0, 1.0 - r * r));
  vec3 v = normalize(p.x * uR + p.y * uU + mu * uC);
  float u = atan(v.y, v.x) / (2.0 * PI);
  float t = acos(clamp(v.z, -1.0, 1.0)) / PI;
  // RA wraps at 0h: take derivatives from whichever unwrapped copy is continuous here
  vec2 a = vec2(fract(u), t), b = vec2(fract(u + 0.5) - 0.5, t);
  vec2 dax = dFdx(a), day = dFdy(a), dbx = dFdx(b), dby = dFdy(b);
  bool useB = abs(dbx.x) + abs(dby.x) < abs(dax.x) + abs(day.x);
  vec3 c = textureGrad(uSky, a, useB ? dbx : dax, useB ? dby : day).rgb;
  c *= 0.78 + 0.22 * mu;
  outColor = vec4(c * cover, cover);
}`;

const SPIN_DEG_PER_S = 5;      // the idle turn: a little over a minute per revolution
const IDLE_MS = 3500;          // after the visitor lets go, the turn resumes
const ZOOM_MIN = 1, ZOOM_MAX = 8;

function shader(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader');
  return s;
}

function mount(root: HTMLElement) {
  const canvas = root.querySelector('canvas')!;
  const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, antialias: false });
  if (!gl) return;

  const prog = gl.createProgram()!;
  gl.attachShader(prog, shader(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, shader(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const U = (n: string) => gl.getUniformLocation(prog, n);
  const uC = U('uC'), uR = U('uR'), uU = U('uU'), uRes = U('uRes'), uZoom = U('uZoom');

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let phi = Number(root.dataset.phi ?? 160);
  let elev = Number(root.dataset.elev ?? 34);
  let zoom = 1;
  let spin = reduced.matches ? 0 : SPIN_DEG_PER_S;  // current rate, eased toward its target
  let lastTouch = -Infinity;
  let visible = false, ready = false, frame = 0, last = 0, dirty = true;
  const pointers = new Map<number, { x: number; y: number }>();
  let dragging = false, pinch0 = 0, zoom0 = 1;

  const img = new Image();
  img.decoding = 'async';
  img.src = root.dataset.texture!;
  img.decode().then(() => {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, 4);
    ready = true;
    draw();
    root.classList.add('live');
    kick();
  }).catch(() => {});

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const n = Math.round(canvas.clientWidth * dpr);
    if (n > 0 && canvas.width !== n) { canvas.width = canvas.height = n; dirty = true; }
  }

  function draw() {
    resize();
    const p = phi * Math.PI / 180, e = elev * Math.PI / 180;
    gl!.viewport(0, 0, canvas.width, canvas.height);
    gl!.uniform3f(uC, Math.cos(e) * Math.cos(p), Math.cos(e) * Math.sin(p), Math.sin(e));
    gl!.uniform3f(uR, Math.sin(p), -Math.cos(p), 0);
    gl!.uniform3f(uU, -Math.sin(e) * Math.cos(p), -Math.sin(e) * Math.sin(p), Math.cos(e));
    gl!.uniform2f(uRes, canvas.width, canvas.height);
    gl!.uniform1f(uZoom, zoom);
    gl!.clearColor(0, 0, 0, 0);
    gl!.clear(gl!.COLOR_BUFFER_BIT);
    gl!.drawArrays(gl!.TRIANGLES, 0, 3);
    dirty = false;
  }

  function tick(now: number) {
    frame = 0;
    const dt = Math.min(0.1, (now - (last || now)) / 1000);
    last = now;
    const target = !reduced.matches && !dragging && now - lastTouch > IDLE_MS ? SPIN_DEG_PER_S : 0;
    spin += (target - spin) * Math.min(1, dt * 1.5);
    if (Math.abs(spin) < 1e-3) spin = 0;
    if (spin) { phi += spin * dt; dirty = true; }
    if (dirty) draw();
    if (spin || target) kick();
    else last = 0;
  }

  function kick() {
    if (!frame && ready && visible) frame = requestAnimationFrame(tick);
  }

  function touched() {
    lastTouch = performance.now();
    dirty = true;
    kick();
  }

  // ---- drag to turn, pinch to zoom

  function setZoom(z: number) {
    zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));
    // at full view a vertical swipe still scrolls the page; zoomed in, the globe takes the touch
    canvas.style.touchAction = zoom > 1.02 ? 'none' : 'pan-y';
    touched();
  }

  function turn(dx: number, dy: number) {
    const k = 180 / Math.PI / (canvas.clientWidth / 2) / zoom;
    phi += dx * k;
    elev = Math.max(-89, Math.min(89, elev + dy * k));
    touched();
  }

  canvas.addEventListener('pointerdown', (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    dragging = true;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch0 = Math.hypot(a.x - b.x, a.y - b.y);
      zoom0 = zoom;
    }
    touched();
  });
  canvas.addEventListener('pointermove', (ev) => {
    const prev = pointers.get(ev.pointerId);
    if (!prev) return;
    const cur = { x: ev.clientX, y: ev.clientY };
    pointers.set(ev.pointerId, cur);
    if (pointers.size === 1) turn(cur.x - prev.x, cur.y - prev.y);
    else if (pointers.size === 2 && pinch0) {
      const [a, b] = [...pointers.values()];
      setZoom(zoom0 * Math.hypot(a.x - b.x, a.y - b.y) / pinch0);
    }
  });
  const release = (ev: PointerEvent) => {
    pointers.delete(ev.pointerId);
    if (pointers.size < 2) pinch0 = 0;
    dragging = pointers.size > 0;
    touched();
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  canvas.addEventListener('wheel', (ev) => {
    if (!(ev.ctrlKey || document.activeElement === canvas)) return;
    ev.preventDefault();
    setZoom(zoom * Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015)));
  }, { passive: false });

  canvas.addEventListener('keydown', (ev) => {
    const step = 40;
    const keys: Record<string, () => void> = {
      ArrowLeft: () => turn(step, 0), ArrowRight: () => turn(-step, 0),
      ArrowUp: () => turn(0, step), ArrowDown: () => turn(0, -step),
      '+': () => setZoom(zoom * 1.25), '=': () => setZoom(zoom * 1.25), '-': () => setZoom(zoom / 1.25),
    };
    if (keys[ev.key]) { ev.preventDefault(); keys[ev.key](); }
  });

  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) { dirty = true; kick(); }
  }).observe(canvas);
  new ResizeObserver(() => { dirty = true; if (ready) draw(); }).observe(canvas);
  reduced.addEventListener('change', () => { dirty = true; kick(); });
}

// Nothing is fetched until the globe is near the viewport.
const lazy = new IntersectionObserver((entries, obs) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    obs.unobserve(e.target);
    mount(e.target as HTMLElement);
  }
}, { rootMargin: '400px' });
document.querySelectorAll<HTMLElement>('[data-globe]').forEach((el) => lazy.observe(el));
