var ae = Object.defineProperty;
var he = (m, t, e) => t in m ? ae(m, t, { enumerable: !0, configurable: !0, writable: !0, value: e }) : m[t] = e;
var b = (m, t, e) => (he(m, typeof t != "symbol" ? t + "" : t, e), e);
var It;
(function(m) {
  m.INIT = "init", m.INIT_CHANNEL = "initChannel", m.INTERMEDIATE = "intermediate", m.EXECUTE = "execute", m.INTERCOM_INIT = "interComInit", m.INTERCOM_INTERMEDIATE = "interComIntermediate", m.INTERCOM_EXECUTE = "interComExecute";
})(It || (It = {}));
var et;
(function(m) {
  m.INIT_COMPLETE = "initComplete", m.INIT_CHANNEL_COMPLETE = "initChannelComplete", m.INTERMEDIATE_CONFIRM = "intermediateConfirm", m.EXECUTE_COMPLETE = "executeComplete", m.INTERCOM_INIT_COMPLETE = "interComInitComplete", m.INTERCOM_INTERMEDIATE_CONFIRM = "interComIntermediateConfirm", m.INTERCOM_EXECUTE_COMPLETE = "interComExecuteComplete";
})(et || (et = {}));
const oe = (m, t) => {
  const e = t.data;
  if (e && e.cmd) {
    const s = m, i = e.cmd ?? "unknown";
    typeof s[i] == "function" ? s[i](e) : console.warn(`No function "${i}" found on workerImpl.`);
  } else
    console.error(`Received: unknown message: ${e}`);
};
var Nt;
(function(m) {
  m.INIT_OFFSCREEN_CANVAS = "initOffscreenCanvas", m.PROXY_START = "proxyStart", m.PROXY_EVENT = "proxyEvent", m.RESIZE = "resize";
})(Nt || (Nt = {}));
var Tt;
(function(m) {
  m.INIT_OFFSCREEN_CANVAS_COMPLETE = "initOffscreenCanvasComplete", m.PROXY_START_COMPLETE = "proxyStartComplete", m.PROXY_EVENT_COMPLETE = "proxyEventComplete", m.RESIZE_COMPLETE = "resizeComplete";
})(Tt || (Tt = {}));
class gt {
}
b(gt, "handler", /* @__PURE__ */ new Map());
class T {
  constructor(t) {
    b(this, "cmd", "unknown");
    b(this, "uuid", "unknown");
    b(this, "name", "unnamed");
    b(this, "workerId", 0);
    b(this, "progress", 0);
    b(this, "payloads", []);
    this.cmd = (t == null ? void 0 : t.cmd) ?? this.cmd, this.name = (t == null ? void 0 : t.name) ?? this.name, this.workerId = (t == null ? void 0 : t.workerId) ?? this.workerId, this.progress = (t == null ? void 0 : t.progress) ?? this.progress;
  }
  addPayload(t) {
    t && (Array.isArray(t) ? this.payloads = this.payloads.concat(t) : this.payloads.push(t));
  }
  static createNew(t) {
    return new T(t);
  }
  static createEmpty() {
    return T.createNew({});
  }
  static createFromExisting(t, e) {
    const s = T.createNew(t);
    return s.uuid = t.uuid, e != null && e.overrideCmd && (s.cmd = e.overrideCmd), e != null && e.overrideUuid && (s.uuid = e.overrideUuid), s;
  }
  static pack(t, e) {
    const s = [];
    if (t)
      for (const i of t) {
        const n = gt.handler.get(i.$type);
        n == null || n.pack(i, s, e === !0);
      }
    return s;
  }
  static unpack(t, e) {
    const s = T.createFromExisting(t, {
      overrideUuid: t.uuid
    });
    if (t.payloads)
      for (const i of t.payloads) {
        const n = gt.handler.get(i.$type);
        s.addPayload(n == null ? void 0 : n.unpack(i, e === !0));
      }
    return s;
  }
  static fromPayload(t, e) {
    const s = T.createNew({
      cmd: e
    });
    return s.addPayload(t), s;
  }
}
const le = (m, t, e) => {
  for (const s of m) {
    const i = e ? s.slice(0) : s, n = i.buffer;
    n ? t.push(n) : t.push(i);
  }
}, Dt = (m, t, e) => {
  if (t)
    for (const [s, i] of Object.entries(t)) {
      const n = "set" + s.substring(0, 1).toLocaleUpperCase() + s.substring(1);
      Object.prototype.hasOwnProperty.call(m, n) && typeof m[n] == "function" ? m[n] = i : (Object.prototype.hasOwnProperty.call(m, s) || e) && (m[s] = i);
    }
};
class Kt {
  constructor() {
    b(this, "$type", "DataPayload");
    b(this, "message", {
      buffers: /* @__PURE__ */ new Map(),
      params: {}
    });
    b(this, "progress", 0);
  }
}
class ue {
  pack(t, e, s) {
    var n;
    const i = t;
    return i.message.buffers && le((n = i.message.buffers) == null ? void 0 : n.values(), e, s), e;
  }
  unpack(t, e) {
    const s = t, i = Object.assign(new Kt(), t);
    if (s.message.buffers)
      for (const [n, h] of s.message.buffers.entries())
        i.message.buffers && i.message.buffers.set(n, e ? h.slice(0) : h);
    return i;
  }
}
gt.handler.set("DataPayload", new ue());
/**
 * @license
 * Copyright 2010-2023 Three.js Authors
 * SPDX-License-Identifier: MIT
 */
const te = "163", ee = 300, Rt = 1e3, ut = 1001, Ot = 1002, Bt = 1003, ce = 1006, de = 1008, me = 1009, pe = 1014, xe = 1020, ge = 1023, ft = 1026, Lt = 1027, se = "", P = "srgb", At = "srgb-linear", ye = "display-p3", ie = "display-p3-linear", zt = "linear", vt = "srgb", Pt = "rec709", Ut = "p3", fe = 515, ct = 2e3, Gt = 2001;
class ne {
  addEventListener(t, e) {
    this._listeners === void 0 && (this._listeners = {});
    const s = this._listeners;
    s[t] === void 0 && (s[t] = []), s[t].indexOf(e) === -1 && s[t].push(e);
  }
  hasEventListener(t, e) {
    if (this._listeners === void 0)
      return !1;
    const s = this._listeners;
    return s[t] !== void 0 && s[t].indexOf(e) !== -1;
  }
  removeEventListener(t, e) {
    if (this._listeners === void 0)
      return;
    const i = this._listeners[t];
    if (i !== void 0) {
      const n = i.indexOf(e);
      n !== -1 && i.splice(n, 1);
    }
  }
  dispatchEvent(t) {
    if (this._listeners === void 0)
      return;
    const s = this._listeners[t.type];
    if (s !== void 0) {
      t.target = this;
      const i = s.slice(0);
      for (let n = 0, h = i.length; n < h; n++)
        i[n].call(this, t);
      t.target = null;
    }
  }
}
const z = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "0a", "0b", "0c", "0d", "0e", "0f", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "1a", "1b", "1c", "1d", "1e", "1f", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "2a", "2b", "2c", "2d", "2e", "2f", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "3a", "3b", "3c", "3d", "3e", "3f", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "4a", "4b", "4c", "4d", "4e", "4f", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5a", "5b", "5c", "5d", "5e", "5f", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "6a", "6b", "6c", "6d", "6e", "6f", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7a", "7b", "7c", "7d", "7e", "7f", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "8a", "8b", "8c", "8d", "8e", "8f", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "9a", "9b", "9c", "9d", "9e", "9f", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "aa", "ab", "ac", "ad", "ae", "af", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "ba", "bb", "bc", "bd", "be", "bf", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "da", "db", "dc", "dd", "de", "df", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "ea", "eb", "ec", "ed", "ee", "ef", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "fa", "fb", "fc", "fd", "fe", "ff"];
function St() {
  const m = Math.random() * 4294967295 | 0, t = Math.random() * 4294967295 | 0, e = Math.random() * 4294967295 | 0, s = Math.random() * 4294967295 | 0;
  return (z[m & 255] + z[m >> 8 & 255] + z[m >> 16 & 255] + z[m >> 24 & 255] + "-" + z[t & 255] + z[t >> 8 & 255] + "-" + z[t >> 16 & 15 | 64] + z[t >> 24 & 255] + "-" + z[e & 63 | 128] + z[e >> 8 & 255] + "-" + z[e >> 16 & 255] + z[e >> 24 & 255] + z[s & 255] + z[s >> 8 & 255] + z[s >> 16 & 255] + z[s >> 24 & 255]).toLowerCase();
}
function S(m, t, e) {
  return Math.max(t, Math.min(e, m));
}
function Me(m, t) {
  return (m % t + t) % t;
}
function Mt(m, t, e) {
  return (1 - e) * m + e * t;
}
class st {
  constructor(t = 0, e = 0) {
    st.prototype.isVector2 = !0, this.x = t, this.y = e;
  }
  get width() {
    return this.x;
  }
  set width(t) {
    this.x = t;
  }
  get height() {
    return this.y;
  }
  set height(t) {
    this.y = t;
  }
  set(t, e) {
    return this.x = t, this.y = e, this;
  }
  setScalar(t) {
    return this.x = t, this.y = t, this;
  }
  setX(t) {
    return this.x = t, this;
  }
  setY(t) {
    return this.y = t, this;
  }
  setComponent(t, e) {
    switch (t) {
      case 0:
        this.x = e;
        break;
      case 1:
        this.y = e;
        break;
      default:
        throw new Error("index is out of range: " + t);
    }
    return this;
  }
  getComponent(t) {
    switch (t) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      default:
        throw new Error("index is out of range: " + t);
    }
  }
  clone() {
    return new this.constructor(this.x, this.y);
  }
  copy(t) {
    return this.x = t.x, this.y = t.y, this;
  }
  add(t) {
    return this.x += t.x, this.y += t.y, this;
  }
  addScalar(t) {
    return this.x += t, this.y += t, this;
  }
  addVectors(t, e) {
    return this.x = t.x + e.x, this.y = t.y + e.y, this;
  }
  addScaledVector(t, e) {
    return this.x += t.x * e, this.y += t.y * e, this;
  }
  sub(t) {
    return this.x -= t.x, this.y -= t.y, this;
  }
  subScalar(t) {
    return this.x -= t, this.y -= t, this;
  }
  subVectors(t, e) {
    return this.x = t.x - e.x, this.y = t.y - e.y, this;
  }
  multiply(t) {
    return this.x *= t.x, this.y *= t.y, this;
  }
  multiplyScalar(t) {
    return this.x *= t, this.y *= t, this;
  }
  divide(t) {
    return this.x /= t.x, this.y /= t.y, this;
  }
  divideScalar(t) {
    return this.multiplyScalar(1 / t);
  }
  applyMatrix3(t) {
    const e = this.x, s = this.y, i = t.elements;
    return this.x = i[0] * e + i[3] * s + i[6], this.y = i[1] * e + i[4] * s + i[7], this;
  }
  min(t) {
    return this.x = Math.min(this.x, t.x), this.y = Math.min(this.y, t.y), this;
  }
  max(t) {
    return this.x = Math.max(this.x, t.x), this.y = Math.max(this.y, t.y), this;
  }
  clamp(t, e) {
    return this.x = Math.max(t.x, Math.min(e.x, this.x)), this.y = Math.max(t.y, Math.min(e.y, this.y)), this;
  }
  clampScalar(t, e) {
    return this.x = Math.max(t, Math.min(e, this.x)), this.y = Math.max(t, Math.min(e, this.y)), this;
  }
  clampLength(t, e) {
    const s = this.length();
    return this.divideScalar(s || 1).multiplyScalar(Math.max(t, Math.min(e, s)));
  }
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this;
  }
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this;
  }
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this;
  }
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this;
  }
  negate() {
    return this.x = -this.x, this.y = -this.y, this;
  }
  dot(t) {
    return this.x * t.x + this.y * t.y;
  }
  cross(t) {
    return this.x * t.y - this.y * t.x;
  }
  lengthSq() {
    return this.x * this.x + this.y * this.y;
  }
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y);
  }
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  angle() {
    return Math.atan2(-this.y, -this.x) + Math.PI;
  }
  angleTo(t) {
    const e = Math.sqrt(this.lengthSq() * t.lengthSq());
    if (e === 0)
      return Math.PI / 2;
    const s = this.dot(t) / e;
    return Math.acos(S(s, -1, 1));
  }
  distanceTo(t) {
    return Math.sqrt(this.distanceToSquared(t));
  }
  distanceToSquared(t) {
    const e = this.x - t.x, s = this.y - t.y;
    return e * e + s * s;
  }
  manhattanDistanceTo(t) {
    return Math.abs(this.x - t.x) + Math.abs(this.y - t.y);
  }
  setLength(t) {
    return this.normalize().multiplyScalar(t);
  }
  lerp(t, e) {
    return this.x += (t.x - this.x) * e, this.y += (t.y - this.y) * e, this;
  }
  lerpVectors(t, e, s) {
    return this.x = t.x + (e.x - t.x) * s, this.y = t.y + (e.y - t.y) * s, this;
  }
  equals(t) {
    return t.x === this.x && t.y === this.y;
  }
  fromArray(t, e = 0) {
    return this.x = t[e], this.y = t[e + 1], this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this.x, t[e + 1] = this.y, t;
  }
  fromBufferAttribute(t, e) {
    return this.x = t.getX(e), this.y = t.getY(e), this;
  }
  rotateAround(t, e) {
    const s = Math.cos(e), i = Math.sin(e), n = this.x - t.x, h = this.y - t.y;
    return this.x = n * s - h * i + t.x, this.y = n * i + h * s + t.y, this;
  }
  random() {
    return this.x = Math.random(), this.y = Math.random(), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y;
  }
}
class j {
  constructor(t, e, s, i, n, h, a, r, o) {
    j.prototype.isMatrix3 = !0, this.elements = [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ], t !== void 0 && this.set(t, e, s, i, n, h, a, r, o);
  }
  set(t, e, s, i, n, h, a, r, o) {
    const l = this.elements;
    return l[0] = t, l[1] = i, l[2] = a, l[3] = e, l[4] = n, l[5] = r, l[6] = s, l[7] = h, l[8] = o, this;
  }
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ), this;
  }
  copy(t) {
    const e = this.elements, s = t.elements;
    return e[0] = s[0], e[1] = s[1], e[2] = s[2], e[3] = s[3], e[4] = s[4], e[5] = s[5], e[6] = s[6], e[7] = s[7], e[8] = s[8], this;
  }
  extractBasis(t, e, s) {
    return t.setFromMatrix3Column(this, 0), e.setFromMatrix3Column(this, 1), s.setFromMatrix3Column(this, 2), this;
  }
  setFromMatrix4(t) {
    const e = t.elements;
    return this.set(
      e[0],
      e[4],
      e[8],
      e[1],
      e[5],
      e[9],
      e[2],
      e[6],
      e[10]
    ), this;
  }
  multiply(t) {
    return this.multiplyMatrices(this, t);
  }
  premultiply(t) {
    return this.multiplyMatrices(t, this);
  }
  multiplyMatrices(t, e) {
    const s = t.elements, i = e.elements, n = this.elements, h = s[0], a = s[3], r = s[6], o = s[1], l = s[4], u = s[7], d = s[2], c = s[5], p = s[8], x = i[0], g = i[3], y = i[6], C = i[1], w = i[4], _ = i[7], M = i[2], F = i[5], f = i[8];
    return n[0] = h * x + a * C + r * M, n[3] = h * g + a * w + r * F, n[6] = h * y + a * _ + r * f, n[1] = o * x + l * C + u * M, n[4] = o * g + l * w + u * F, n[7] = o * y + l * _ + u * f, n[2] = d * x + c * C + p * M, n[5] = d * g + c * w + p * F, n[8] = d * y + c * _ + p * f, this;
  }
  multiplyScalar(t) {
    const e = this.elements;
    return e[0] *= t, e[3] *= t, e[6] *= t, e[1] *= t, e[4] *= t, e[7] *= t, e[2] *= t, e[5] *= t, e[8] *= t, this;
  }
  determinant() {
    const t = this.elements, e = t[0], s = t[1], i = t[2], n = t[3], h = t[4], a = t[5], r = t[6], o = t[7], l = t[8];
    return e * h * l - e * a * o - s * n * l + s * a * r + i * n * o - i * h * r;
  }
  invert() {
    const t = this.elements, e = t[0], s = t[1], i = t[2], n = t[3], h = t[4], a = t[5], r = t[6], o = t[7], l = t[8], u = l * h - a * o, d = a * r - l * n, c = o * n - h * r, p = e * u + s * d + i * c;
    if (p === 0)
      return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0);
    const x = 1 / p;
    return t[0] = u * x, t[1] = (i * o - l * s) * x, t[2] = (a * s - i * h) * x, t[3] = d * x, t[4] = (l * e - i * r) * x, t[5] = (i * n - a * e) * x, t[6] = c * x, t[7] = (s * r - o * e) * x, t[8] = (h * e - s * n) * x, this;
  }
  transpose() {
    let t;
    const e = this.elements;
    return t = e[1], e[1] = e[3], e[3] = t, t = e[2], e[2] = e[6], e[6] = t, t = e[5], e[5] = e[7], e[7] = t, this;
  }
  getNormalMatrix(t) {
    return this.setFromMatrix4(t).invert().transpose();
  }
  transposeIntoArray(t) {
    const e = this.elements;
    return t[0] = e[0], t[1] = e[3], t[2] = e[6], t[3] = e[1], t[4] = e[4], t[5] = e[7], t[6] = e[2], t[7] = e[5], t[8] = e[8], this;
  }
  setUvTransform(t, e, s, i, n, h, a) {
    const r = Math.cos(n), o = Math.sin(n);
    return this.set(
      s * r,
      s * o,
      -s * (r * h + o * a) + h + t,
      -i * o,
      i * r,
      -i * (-o * h + r * a) + a + e,
      0,
      0,
      1
    ), this;
  }
  //
  scale(t, e) {
    return this.premultiply(bt.makeScale(t, e)), this;
  }
  rotate(t) {
    return this.premultiply(bt.makeRotation(-t)), this;
  }
  translate(t, e) {
    return this.premultiply(bt.makeTranslation(t, e)), this;
  }
  // for 2D Transforms
  makeTranslation(t, e) {
    return t.isVector2 ? this.set(
      1,
      0,
      t.x,
      0,
      1,
      t.y,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      t,
      0,
      1,
      e,
      0,
      0,
      1
    ), this;
  }
  makeRotation(t) {
    const e = Math.cos(t), s = Math.sin(t);
    return this.set(
      e,
      -s,
      0,
      s,
      e,
      0,
      0,
      0,
      1
    ), this;
  }
  makeScale(t, e) {
    return this.set(
      t,
      0,
      0,
      0,
      e,
      0,
      0,
      0,
      1
    ), this;
  }
  //
  equals(t) {
    const e = this.elements, s = t.elements;
    for (let i = 0; i < 9; i++)
      if (e[i] !== s[i])
        return !1;
    return !0;
  }
  fromArray(t, e = 0) {
    for (let s = 0; s < 9; s++)
      this.elements[s] = t[s + e];
    return this;
  }
  toArray(t = [], e = 0) {
    const s = this.elements;
    return t[e] = s[0], t[e + 1] = s[1], t[e + 2] = s[2], t[e + 3] = s[3], t[e + 4] = s[4], t[e + 5] = s[5], t[e + 6] = s[6], t[e + 7] = s[7], t[e + 8] = s[8], t;
  }
  clone() {
    return new this.constructor().fromArray(this.elements);
  }
}
const bt = /* @__PURE__ */ new j();
function Wt(m) {
  return document.createElementNS("http://www.w3.org/1999/xhtml", m);
}
const qt = /* @__PURE__ */ new j().set(
  0.8224621,
  0.177538,
  0,
  0.0331941,
  0.9668058,
  0,
  0.0170827,
  0.0723974,
  0.9105199
), Ht = /* @__PURE__ */ new j().set(
  1.2249401,
  -0.2249404,
  0,
  -0.0420569,
  1.0420571,
  0,
  -0.0196376,
  -0.0786361,
  1.0982735
), dt = {
  [At]: {
    transfer: zt,
    primaries: Pt,
    toReference: (m) => m,
    fromReference: (m) => m
  },
  [P]: {
    transfer: vt,
    primaries: Pt,
    toReference: (m) => m.convertSRGBToLinear(),
    fromReference: (m) => m.convertLinearToSRGB()
  },
  [ie]: {
    transfer: zt,
    primaries: Ut,
    toReference: (m) => m.applyMatrix3(Ht),
    fromReference: (m) => m.applyMatrix3(qt)
  },
  [ye]: {
    transfer: vt,
    primaries: Ut,
    toReference: (m) => m.convertSRGBToLinear().applyMatrix3(Ht),
    fromReference: (m) => m.applyMatrix3(qt).convertLinearToSRGB()
  }
}, be = /* @__PURE__ */ new Set([At, ie]), I = {
  enabled: !0,
  _workingColorSpace: At,
  get workingColorSpace() {
    return this._workingColorSpace;
  },
  set workingColorSpace(m) {
    if (!be.has(m))
      throw new Error(`Unsupported working color space, "${m}".`);
    this._workingColorSpace = m;
  },
  convert: function(m, t, e) {
    if (this.enabled === !1 || t === e || !t || !e)
      return m;
    const s = dt[t].toReference, i = dt[e].fromReference;
    return i(s(m));
  },
  fromWorkingColorSpace: function(m, t) {
    return this.convert(m, this._workingColorSpace, t);
  },
  toWorkingColorSpace: function(m, t) {
    return this.convert(m, t, this._workingColorSpace);
  },
  getPrimaries: function(m) {
    return dt[m].primaries;
  },
  getTransfer: function(m) {
    return m === se ? zt : dt[m].transfer;
  }
};
function K(m) {
  return m < 0.04045 ? m * 0.0773993808 : Math.pow(m * 0.9478672986 + 0.0521327014, 2.4);
}
function _t(m) {
  return m < 31308e-7 ? m * 12.92 : 1.055 * Math.pow(m, 0.41666) - 0.055;
}
let X;
class _e {
  static getDataURL(t) {
    if (/^data:/i.test(t.src) || typeof HTMLCanvasElement > "u")
      return t.src;
    let e;
    if (t instanceof HTMLCanvasElement)
      e = t;
    else {
      X === void 0 && (X = Wt("canvas")), X.width = t.width, X.height = t.height;
      const s = X.getContext("2d");
      t instanceof ImageData ? s.putImageData(t, 0, 0) : s.drawImage(t, 0, 0, t.width, t.height), e = X;
    }
    return e.width > 2048 || e.height > 2048 ? (console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons", t), e.toDataURL("image/jpeg", 0.6)) : e.toDataURL("image/png");
  }
  static sRGBToLinear(t) {
    if (typeof HTMLImageElement < "u" && t instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && t instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && t instanceof ImageBitmap) {
      const e = Wt("canvas");
      e.width = t.width, e.height = t.height;
      const s = e.getContext("2d");
      s.drawImage(t, 0, 0, t.width, t.height);
      const i = s.getImageData(0, 0, t.width, t.height), n = i.data;
      for (let h = 0; h < n.length; h++)
        n[h] = K(n[h] / 255) * 255;
      return s.putImageData(i, 0, 0), e;
    } else if (t.data) {
      const e = t.data.slice(0);
      for (let s = 0; s < e.length; s++)
        e instanceof Uint8Array || e instanceof Uint8ClampedArray ? e[s] = Math.floor(K(e[s] / 255) * 255) : e[s] = K(e[s]);
      return {
        data: e,
        width: t.width,
        height: t.height
      };
    } else
      return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."), t;
  }
}
let we = 0;
class Fe {
  constructor(t = null) {
    this.isSource = !0, Object.defineProperty(this, "id", { value: we++ }), this.uuid = St(), this.data = t, this.dataReady = !0, this.version = 0;
  }
  set needsUpdate(t) {
    t === !0 && this.version++;
  }
  toJSON(t) {
    const e = t === void 0 || typeof t == "string";
    if (!e && t.images[this.uuid] !== void 0)
      return t.images[this.uuid];
    const s = {
      uuid: this.uuid,
      url: ""
    }, i = this.data;
    if (i !== null) {
      let n;
      if (Array.isArray(i)) {
        n = [];
        for (let h = 0, a = i.length; h < a; h++)
          i[h].isDataTexture ? n.push(wt(i[h].image)) : n.push(wt(i[h]));
      } else
        n = wt(i);
      s.url = n;
    }
    return e || (t.images[this.uuid] = s), s;
  }
}
function wt(m) {
  return typeof HTMLImageElement < "u" && m instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && m instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && m instanceof ImageBitmap ? _e.getDataURL(m) : m.data ? {
    data: Array.from(m.data),
    width: m.width,
    height: m.height,
    type: m.data.constructor.name
  } : (console.warn("THREE.Texture: Unable to serialize Texture."), {});
}
let Ce = 0;
class U extends ne {
  constructor(t = U.DEFAULT_IMAGE, e = U.DEFAULT_MAPPING, s = ut, i = ut, n = ce, h = de, a = ge, r = me, o = U.DEFAULT_ANISOTROPY, l = se) {
    super(), this.isTexture = !0, Object.defineProperty(this, "id", { value: Ce++ }), this.uuid = St(), this.name = "", this.source = new Fe(t), this.mipmaps = [], this.mapping = e, this.channel = 0, this.wrapS = s, this.wrapT = i, this.magFilter = n, this.minFilter = h, this.anisotropy = o, this.format = a, this.internalFormat = null, this.type = r, this.offset = new st(0, 0), this.repeat = new st(1, 1), this.center = new st(0, 0), this.rotation = 0, this.matrixAutoUpdate = !0, this.matrix = new j(), this.generateMipmaps = !0, this.premultiplyAlpha = !1, this.flipY = !0, this.unpackAlignment = 4, this.colorSpace = l, this.userData = {}, this.version = 0, this.onUpdate = null, this.isRenderTargetTexture = !1, this.pmremVersion = 0;
  }
  get image() {
    return this.source.data;
  }
  set image(t = null) {
    this.source.data = t;
  }
  updateMatrix() {
    this.matrix.setUvTransform(this.offset.x, this.offset.y, this.repeat.x, this.repeat.y, this.rotation, this.center.x, this.center.y);
  }
  clone() {
    return new this.constructor().copy(this);
  }
  copy(t) {
    return this.name = t.name, this.source = t.source, this.mipmaps = t.mipmaps.slice(0), this.mapping = t.mapping, this.channel = t.channel, this.wrapS = t.wrapS, this.wrapT = t.wrapT, this.magFilter = t.magFilter, this.minFilter = t.minFilter, this.anisotropy = t.anisotropy, this.format = t.format, this.internalFormat = t.internalFormat, this.type = t.type, this.offset.copy(t.offset), this.repeat.copy(t.repeat), this.center.copy(t.center), this.rotation = t.rotation, this.matrixAutoUpdate = t.matrixAutoUpdate, this.matrix.copy(t.matrix), this.generateMipmaps = t.generateMipmaps, this.premultiplyAlpha = t.premultiplyAlpha, this.flipY = t.flipY, this.unpackAlignment = t.unpackAlignment, this.colorSpace = t.colorSpace, this.userData = JSON.parse(JSON.stringify(t.userData)), this.needsUpdate = !0, this;
  }
  toJSON(t) {
    const e = t === void 0 || typeof t == "string";
    if (!e && t.textures[this.uuid] !== void 0)
      return t.textures[this.uuid];
    const s = {
      metadata: {
        version: 4.6,
        type: "Texture",
        generator: "Texture.toJSON"
      },
      uuid: this.uuid,
      name: this.name,
      image: this.source.toJSON(t).uuid,
      mapping: this.mapping,
      channel: this.channel,
      repeat: [this.repeat.x, this.repeat.y],
      offset: [this.offset.x, this.offset.y],
      center: [this.center.x, this.center.y],
      rotation: this.rotation,
      wrap: [this.wrapS, this.wrapT],
      format: this.format,
      internalFormat: this.internalFormat,
      type: this.type,
      colorSpace: this.colorSpace,
      minFilter: this.minFilter,
      magFilter: this.magFilter,
      anisotropy: this.anisotropy,
      flipY: this.flipY,
      generateMipmaps: this.generateMipmaps,
      premultiplyAlpha: this.premultiplyAlpha,
      unpackAlignment: this.unpackAlignment
    };
    return Object.keys(this.userData).length > 0 && (s.userData = this.userData), e || (t.textures[this.uuid] = s), s;
  }
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
  transformUv(t) {
    if (this.mapping !== ee)
      return t;
    if (t.applyMatrix3(this.matrix), t.x < 0 || t.x > 1)
      switch (this.wrapS) {
        case Rt:
          t.x = t.x - Math.floor(t.x);
          break;
        case ut:
          t.x = t.x < 0 ? 0 : 1;
          break;
        case Ot:
          Math.abs(Math.floor(t.x) % 2) === 1 ? t.x = Math.ceil(t.x) - t.x : t.x = t.x - Math.floor(t.x);
          break;
      }
    if (t.y < 0 || t.y > 1)
      switch (this.wrapT) {
        case Rt:
          t.y = t.y - Math.floor(t.y);
          break;
        case ut:
          t.y = t.y < 0 ? 0 : 1;
          break;
        case Ot:
          Math.abs(Math.floor(t.y) % 2) === 1 ? t.y = Math.ceil(t.y) - t.y : t.y = t.y - Math.floor(t.y);
          break;
      }
    return this.flipY && (t.y = 1 - t.y), t;
  }
  set needsUpdate(t) {
    t === !0 && (this.version++, this.source.needsUpdate = !0);
  }
  set needsPMREMUpdate(t) {
    t === !0 && this.pmremVersion++;
  }
}
U.DEFAULT_IMAGE = null;
U.DEFAULT_MAPPING = ee;
U.DEFAULT_ANISOTROPY = 1;
class it {
  constructor(t = 0, e = 0, s = 0, i = 1) {
    this.isQuaternion = !0, this._x = t, this._y = e, this._z = s, this._w = i;
  }
  static slerpFlat(t, e, s, i, n, h, a) {
    let r = s[i + 0], o = s[i + 1], l = s[i + 2], u = s[i + 3];
    const d = n[h + 0], c = n[h + 1], p = n[h + 2], x = n[h + 3];
    if (a === 0) {
      t[e + 0] = r, t[e + 1] = o, t[e + 2] = l, t[e + 3] = u;
      return;
    }
    if (a === 1) {
      t[e + 0] = d, t[e + 1] = c, t[e + 2] = p, t[e + 3] = x;
      return;
    }
    if (u !== x || r !== d || o !== c || l !== p) {
      let g = 1 - a;
      const y = r * d + o * c + l * p + u * x, C = y >= 0 ? 1 : -1, w = 1 - y * y;
      if (w > Number.EPSILON) {
        const M = Math.sqrt(w), F = Math.atan2(M, y * C);
        g = Math.sin(g * F) / M, a = Math.sin(a * F) / M;
      }
      const _ = a * C;
      if (r = r * g + d * _, o = o * g + c * _, l = l * g + p * _, u = u * g + x * _, g === 1 - a) {
        const M = 1 / Math.sqrt(r * r + o * o + l * l + u * u);
        r *= M, o *= M, l *= M, u *= M;
      }
    }
    t[e] = r, t[e + 1] = o, t[e + 2] = l, t[e + 3] = u;
  }
  static multiplyQuaternionsFlat(t, e, s, i, n, h) {
    const a = s[i], r = s[i + 1], o = s[i + 2], l = s[i + 3], u = n[h], d = n[h + 1], c = n[h + 2], p = n[h + 3];
    return t[e] = a * p + l * u + r * c - o * d, t[e + 1] = r * p + l * d + o * u - a * c, t[e + 2] = o * p + l * c + a * d - r * u, t[e + 3] = l * p - a * u - r * d - o * c, t;
  }
  get x() {
    return this._x;
  }
  set x(t) {
    this._x = t, this._onChangeCallback();
  }
  get y() {
    return this._y;
  }
  set y(t) {
    this._y = t, this._onChangeCallback();
  }
  get z() {
    return this._z;
  }
  set z(t) {
    this._z = t, this._onChangeCallback();
  }
  get w() {
    return this._w;
  }
  set w(t) {
    this._w = t, this._onChangeCallback();
  }
  set(t, e, s, i) {
    return this._x = t, this._y = e, this._z = s, this._w = i, this._onChangeCallback(), this;
  }
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._w);
  }
  copy(t) {
    return this._x = t.x, this._y = t.y, this._z = t.z, this._w = t.w, this._onChangeCallback(), this;
  }
  setFromEuler(t, e = !0) {
    const s = t._x, i = t._y, n = t._z, h = t._order, a = Math.cos, r = Math.sin, o = a(s / 2), l = a(i / 2), u = a(n / 2), d = r(s / 2), c = r(i / 2), p = r(n / 2);
    switch (h) {
      case "XYZ":
        this._x = d * l * u + o * c * p, this._y = o * c * u - d * l * p, this._z = o * l * p + d * c * u, this._w = o * l * u - d * c * p;
        break;
      case "YXZ":
        this._x = d * l * u + o * c * p, this._y = o * c * u - d * l * p, this._z = o * l * p - d * c * u, this._w = o * l * u + d * c * p;
        break;
      case "ZXY":
        this._x = d * l * u - o * c * p, this._y = o * c * u + d * l * p, this._z = o * l * p + d * c * u, this._w = o * l * u - d * c * p;
        break;
      case "ZYX":
        this._x = d * l * u - o * c * p, this._y = o * c * u + d * l * p, this._z = o * l * p - d * c * u, this._w = o * l * u + d * c * p;
        break;
      case "YZX":
        this._x = d * l * u + o * c * p, this._y = o * c * u + d * l * p, this._z = o * l * p - d * c * u, this._w = o * l * u - d * c * p;
        break;
      case "XZY":
        this._x = d * l * u - o * c * p, this._y = o * c * u - d * l * p, this._z = o * l * p + d * c * u, this._w = o * l * u + d * c * p;
        break;
      default:
        console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: " + h);
    }
    return e === !0 && this._onChangeCallback(), this;
  }
  setFromAxisAngle(t, e) {
    const s = e / 2, i = Math.sin(s);
    return this._x = t.x * i, this._y = t.y * i, this._z = t.z * i, this._w = Math.cos(s), this._onChangeCallback(), this;
  }
  setFromRotationMatrix(t) {
    const e = t.elements, s = e[0], i = e[4], n = e[8], h = e[1], a = e[5], r = e[9], o = e[2], l = e[6], u = e[10], d = s + a + u;
    if (d > 0) {
      const c = 0.5 / Math.sqrt(d + 1);
      this._w = 0.25 / c, this._x = (l - r) * c, this._y = (n - o) * c, this._z = (h - i) * c;
    } else if (s > a && s > u) {
      const c = 2 * Math.sqrt(1 + s - a - u);
      this._w = (l - r) / c, this._x = 0.25 * c, this._y = (i + h) / c, this._z = (n + o) / c;
    } else if (a > u) {
      const c = 2 * Math.sqrt(1 + a - s - u);
      this._w = (n - o) / c, this._x = (i + h) / c, this._y = 0.25 * c, this._z = (r + l) / c;
    } else {
      const c = 2 * Math.sqrt(1 + u - s - a);
      this._w = (h - i) / c, this._x = (n + o) / c, this._y = (r + l) / c, this._z = 0.25 * c;
    }
    return this._onChangeCallback(), this;
  }
  setFromUnitVectors(t, e) {
    let s = t.dot(e) + 1;
    return s < Number.EPSILON ? (s = 0, Math.abs(t.x) > Math.abs(t.z) ? (this._x = -t.y, this._y = t.x, this._z = 0, this._w = s) : (this._x = 0, this._y = -t.z, this._z = t.y, this._w = s)) : (this._x = t.y * e.z - t.z * e.y, this._y = t.z * e.x - t.x * e.z, this._z = t.x * e.y - t.y * e.x, this._w = s), this.normalize();
  }
  angleTo(t) {
    return 2 * Math.acos(Math.abs(S(this.dot(t), -1, 1)));
  }
  rotateTowards(t, e) {
    const s = this.angleTo(t);
    if (s === 0)
      return this;
    const i = Math.min(1, e / s);
    return this.slerp(t, i), this;
  }
  identity() {
    return this.set(0, 0, 0, 1);
  }
  invert() {
    return this.conjugate();
  }
  conjugate() {
    return this._x *= -1, this._y *= -1, this._z *= -1, this._onChangeCallback(), this;
  }
  dot(t) {
    return this._x * t._x + this._y * t._y + this._z * t._z + this._w * t._w;
  }
  lengthSq() {
    return this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w;
  }
  length() {
    return Math.sqrt(this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w);
  }
  normalize() {
    let t = this.length();
    return t === 0 ? (this._x = 0, this._y = 0, this._z = 0, this._w = 1) : (t = 1 / t, this._x = this._x * t, this._y = this._y * t, this._z = this._z * t, this._w = this._w * t), this._onChangeCallback(), this;
  }
  multiply(t) {
    return this.multiplyQuaternions(this, t);
  }
  premultiply(t) {
    return this.multiplyQuaternions(t, this);
  }
  multiplyQuaternions(t, e) {
    const s = t._x, i = t._y, n = t._z, h = t._w, a = e._x, r = e._y, o = e._z, l = e._w;
    return this._x = s * l + h * a + i * o - n * r, this._y = i * l + h * r + n * a - s * o, this._z = n * l + h * o + s * r - i * a, this._w = h * l - s * a - i * r - n * o, this._onChangeCallback(), this;
  }
  slerp(t, e) {
    if (e === 0)
      return this;
    if (e === 1)
      return this.copy(t);
    const s = this._x, i = this._y, n = this._z, h = this._w;
    let a = h * t._w + s * t._x + i * t._y + n * t._z;
    if (a < 0 ? (this._w = -t._w, this._x = -t._x, this._y = -t._y, this._z = -t._z, a = -a) : this.copy(t), a >= 1)
      return this._w = h, this._x = s, this._y = i, this._z = n, this;
    const r = 1 - a * a;
    if (r <= Number.EPSILON) {
      const c = 1 - e;
      return this._w = c * h + e * this._w, this._x = c * s + e * this._x, this._y = c * i + e * this._y, this._z = c * n + e * this._z, this.normalize(), this;
    }
    const o = Math.sqrt(r), l = Math.atan2(o, a), u = Math.sin((1 - e) * l) / o, d = Math.sin(e * l) / o;
    return this._w = h * u + this._w * d, this._x = s * u + this._x * d, this._y = i * u + this._y * d, this._z = n * u + this._z * d, this._onChangeCallback(), this;
  }
  slerpQuaternions(t, e, s) {
    return this.copy(t).slerp(e, s);
  }
  random() {
    const t = 2 * Math.PI * Math.random(), e = 2 * Math.PI * Math.random(), s = Math.random(), i = Math.sqrt(1 - s), n = Math.sqrt(s);
    return this.set(
      i * Math.sin(t),
      i * Math.cos(t),
      n * Math.sin(e),
      n * Math.cos(e)
    );
  }
  equals(t) {
    return t._x === this._x && t._y === this._y && t._z === this._z && t._w === this._w;
  }
  fromArray(t, e = 0) {
    return this._x = t[e], this._y = t[e + 1], this._z = t[e + 2], this._w = t[e + 3], this._onChangeCallback(), this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this._x, t[e + 1] = this._y, t[e + 2] = this._z, t[e + 3] = this._w, t;
  }
  fromBufferAttribute(t, e) {
    return this._x = t.getX(e), this._y = t.getY(e), this._z = t.getZ(e), this._w = t.getW(e), this._onChangeCallback(), this;
  }
  toJSON() {
    return this.toArray();
  }
  _onChange(t) {
    return this._onChangeCallback = t, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._w;
  }
}
class E {
  constructor(t = 0, e = 0, s = 0) {
    E.prototype.isVector3 = !0, this.x = t, this.y = e, this.z = s;
  }
  set(t, e, s) {
    return s === void 0 && (s = this.z), this.x = t, this.y = e, this.z = s, this;
  }
  setScalar(t) {
    return this.x = t, this.y = t, this.z = t, this;
  }
  setX(t) {
    return this.x = t, this;
  }
  setY(t) {
    return this.y = t, this;
  }
  setZ(t) {
    return this.z = t, this;
  }
  setComponent(t, e) {
    switch (t) {
      case 0:
        this.x = e;
        break;
      case 1:
        this.y = e;
        break;
      case 2:
        this.z = e;
        break;
      default:
        throw new Error("index is out of range: " + t);
    }
    return this;
  }
  getComponent(t) {
    switch (t) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      default:
        throw new Error("index is out of range: " + t);
    }
  }
  clone() {
    return new this.constructor(this.x, this.y, this.z);
  }
  copy(t) {
    return this.x = t.x, this.y = t.y, this.z = t.z, this;
  }
  add(t) {
    return this.x += t.x, this.y += t.y, this.z += t.z, this;
  }
  addScalar(t) {
    return this.x += t, this.y += t, this.z += t, this;
  }
  addVectors(t, e) {
    return this.x = t.x + e.x, this.y = t.y + e.y, this.z = t.z + e.z, this;
  }
  addScaledVector(t, e) {
    return this.x += t.x * e, this.y += t.y * e, this.z += t.z * e, this;
  }
  sub(t) {
    return this.x -= t.x, this.y -= t.y, this.z -= t.z, this;
  }
  subScalar(t) {
    return this.x -= t, this.y -= t, this.z -= t, this;
  }
  subVectors(t, e) {
    return this.x = t.x - e.x, this.y = t.y - e.y, this.z = t.z - e.z, this;
  }
  multiply(t) {
    return this.x *= t.x, this.y *= t.y, this.z *= t.z, this;
  }
  multiplyScalar(t) {
    return this.x *= t, this.y *= t, this.z *= t, this;
  }
  multiplyVectors(t, e) {
    return this.x = t.x * e.x, this.y = t.y * e.y, this.z = t.z * e.z, this;
  }
  applyEuler(t) {
    return this.applyQuaternion(jt.setFromEuler(t));
  }
  applyAxisAngle(t, e) {
    return this.applyQuaternion(jt.setFromAxisAngle(t, e));
  }
  applyMatrix3(t) {
    const e = this.x, s = this.y, i = this.z, n = t.elements;
    return this.x = n[0] * e + n[3] * s + n[6] * i, this.y = n[1] * e + n[4] * s + n[7] * i, this.z = n[2] * e + n[5] * s + n[8] * i, this;
  }
  applyNormalMatrix(t) {
    return this.applyMatrix3(t).normalize();
  }
  applyMatrix4(t) {
    const e = this.x, s = this.y, i = this.z, n = t.elements, h = 1 / (n[3] * e + n[7] * s + n[11] * i + n[15]);
    return this.x = (n[0] * e + n[4] * s + n[8] * i + n[12]) * h, this.y = (n[1] * e + n[5] * s + n[9] * i + n[13]) * h, this.z = (n[2] * e + n[6] * s + n[10] * i + n[14]) * h, this;
  }
  applyQuaternion(t) {
    const e = this.x, s = this.y, i = this.z, n = t.x, h = t.y, a = t.z, r = t.w, o = 2 * (h * i - a * s), l = 2 * (a * e - n * i), u = 2 * (n * s - h * e);
    return this.x = e + r * o + h * u - a * l, this.y = s + r * l + a * o - n * u, this.z = i + r * u + n * l - h * o, this;
  }
  project(t) {
    return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix);
  }
  unproject(t) {
    return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld);
  }
  transformDirection(t) {
    const e = this.x, s = this.y, i = this.z, n = t.elements;
    return this.x = n[0] * e + n[4] * s + n[8] * i, this.y = n[1] * e + n[5] * s + n[9] * i, this.z = n[2] * e + n[6] * s + n[10] * i, this.normalize();
  }
  divide(t) {
    return this.x /= t.x, this.y /= t.y, this.z /= t.z, this;
  }
  divideScalar(t) {
    return this.multiplyScalar(1 / t);
  }
  min(t) {
    return this.x = Math.min(this.x, t.x), this.y = Math.min(this.y, t.y), this.z = Math.min(this.z, t.z), this;
  }
  max(t) {
    return this.x = Math.max(this.x, t.x), this.y = Math.max(this.y, t.y), this.z = Math.max(this.z, t.z), this;
  }
  clamp(t, e) {
    return this.x = Math.max(t.x, Math.min(e.x, this.x)), this.y = Math.max(t.y, Math.min(e.y, this.y)), this.z = Math.max(t.z, Math.min(e.z, this.z)), this;
  }
  clampScalar(t, e) {
    return this.x = Math.max(t, Math.min(e, this.x)), this.y = Math.max(t, Math.min(e, this.y)), this.z = Math.max(t, Math.min(e, this.z)), this;
  }
  clampLength(t, e) {
    const s = this.length();
    return this.divideScalar(s || 1).multiplyScalar(Math.max(t, Math.min(e, s)));
  }
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this;
  }
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this;
  }
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this;
  }
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this;
  }
  negate() {
    return this.x = -this.x, this.y = -this.y, this.z = -this.z, this;
  }
  dot(t) {
    return this.x * t.x + this.y * t.y + this.z * t.z;
  }
  // TODO lengthSquared?
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z);
  }
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  setLength(t) {
    return this.normalize().multiplyScalar(t);
  }
  lerp(t, e) {
    return this.x += (t.x - this.x) * e, this.y += (t.y - this.y) * e, this.z += (t.z - this.z) * e, this;
  }
  lerpVectors(t, e, s) {
    return this.x = t.x + (e.x - t.x) * s, this.y = t.y + (e.y - t.y) * s, this.z = t.z + (e.z - t.z) * s, this;
  }
  cross(t) {
    return this.crossVectors(this, t);
  }
  crossVectors(t, e) {
    const s = t.x, i = t.y, n = t.z, h = e.x, a = e.y, r = e.z;
    return this.x = i * r - n * a, this.y = n * h - s * r, this.z = s * a - i * h, this;
  }
  projectOnVector(t) {
    const e = t.lengthSq();
    if (e === 0)
      return this.set(0, 0, 0);
    const s = t.dot(this) / e;
    return this.copy(t).multiplyScalar(s);
  }
  projectOnPlane(t) {
    return Ft.copy(this).projectOnVector(t), this.sub(Ft);
  }
  reflect(t) {
    return this.sub(Ft.copy(t).multiplyScalar(2 * this.dot(t)));
  }
  angleTo(t) {
    const e = Math.sqrt(this.lengthSq() * t.lengthSq());
    if (e === 0)
      return Math.PI / 2;
    const s = this.dot(t) / e;
    return Math.acos(S(s, -1, 1));
  }
  distanceTo(t) {
    return Math.sqrt(this.distanceToSquared(t));
  }
  distanceToSquared(t) {
    const e = this.x - t.x, s = this.y - t.y, i = this.z - t.z;
    return e * e + s * s + i * i;
  }
  manhattanDistanceTo(t) {
    return Math.abs(this.x - t.x) + Math.abs(this.y - t.y) + Math.abs(this.z - t.z);
  }
  setFromSpherical(t) {
    return this.setFromSphericalCoords(t.radius, t.phi, t.theta);
  }
  setFromSphericalCoords(t, e, s) {
    const i = Math.sin(e) * t;
    return this.x = i * Math.sin(s), this.y = Math.cos(e) * t, this.z = i * Math.cos(s), this;
  }
  setFromCylindrical(t) {
    return this.setFromCylindricalCoords(t.radius, t.theta, t.y);
  }
  setFromCylindricalCoords(t, e, s) {
    return this.x = t * Math.sin(e), this.y = s, this.z = t * Math.cos(e), this;
  }
  setFromMatrixPosition(t) {
    const e = t.elements;
    return this.x = e[12], this.y = e[13], this.z = e[14], this;
  }
  setFromMatrixScale(t) {
    const e = this.setFromMatrixColumn(t, 0).length(), s = this.setFromMatrixColumn(t, 1).length(), i = this.setFromMatrixColumn(t, 2).length();
    return this.x = e, this.y = s, this.z = i, this;
  }
  setFromMatrixColumn(t, e) {
    return this.fromArray(t.elements, e * 4);
  }
  setFromMatrix3Column(t, e) {
    return this.fromArray(t.elements, e * 3);
  }
  setFromEuler(t) {
    return this.x = t._x, this.y = t._y, this.z = t._z, this;
  }
  setFromColor(t) {
    return this.x = t.r, this.y = t.g, this.z = t.b, this;
  }
  equals(t) {
    return t.x === this.x && t.y === this.y && t.z === this.z;
  }
  fromArray(t, e = 0) {
    return this.x = t[e], this.y = t[e + 1], this.z = t[e + 2], this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this.x, t[e + 1] = this.y, t[e + 2] = this.z, t;
  }
  fromBufferAttribute(t, e) {
    return this.x = t.getX(e), this.y = t.getY(e), this.z = t.getZ(e), this;
  }
  random() {
    return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this;
  }
  randomDirection() {
    const t = Math.random() * Math.PI * 2, e = Math.random() * 2 - 1, s = Math.sqrt(1 - e * e);
    return this.x = s * Math.cos(t), this.y = e, this.z = s * Math.sin(t), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y, yield this.z;
  }
}
const Ft = /* @__PURE__ */ new E(), jt = /* @__PURE__ */ new it();
class B {
  constructor(t, e, s, i, n, h, a, r, o, l, u, d, c, p, x, g) {
    B.prototype.isMatrix4 = !0, this.elements = [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ], t !== void 0 && this.set(t, e, s, i, n, h, a, r, o, l, u, d, c, p, x, g);
  }
  set(t, e, s, i, n, h, a, r, o, l, u, d, c, p, x, g) {
    const y = this.elements;
    return y[0] = t, y[4] = e, y[8] = s, y[12] = i, y[1] = n, y[5] = h, y[9] = a, y[13] = r, y[2] = o, y[6] = l, y[10] = u, y[14] = d, y[3] = c, y[7] = p, y[11] = x, y[15] = g, this;
  }
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  clone() {
    return new B().fromArray(this.elements);
  }
  copy(t) {
    const e = this.elements, s = t.elements;
    return e[0] = s[0], e[1] = s[1], e[2] = s[2], e[3] = s[3], e[4] = s[4], e[5] = s[5], e[6] = s[6], e[7] = s[7], e[8] = s[8], e[9] = s[9], e[10] = s[10], e[11] = s[11], e[12] = s[12], e[13] = s[13], e[14] = s[14], e[15] = s[15], this;
  }
  copyPosition(t) {
    const e = this.elements, s = t.elements;
    return e[12] = s[12], e[13] = s[13], e[14] = s[14], this;
  }
  setFromMatrix3(t) {
    const e = t.elements;
    return this.set(
      e[0],
      e[3],
      e[6],
      0,
      e[1],
      e[4],
      e[7],
      0,
      e[2],
      e[5],
      e[8],
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  extractBasis(t, e, s) {
    return t.setFromMatrixColumn(this, 0), e.setFromMatrixColumn(this, 1), s.setFromMatrixColumn(this, 2), this;
  }
  makeBasis(t, e, s) {
    return this.set(
      t.x,
      e.x,
      s.x,
      0,
      t.y,
      e.y,
      s.y,
      0,
      t.z,
      e.z,
      s.z,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  extractRotation(t) {
    const e = this.elements, s = t.elements, i = 1 / Z.setFromMatrixColumn(t, 0).length(), n = 1 / Z.setFromMatrixColumn(t, 1).length(), h = 1 / Z.setFromMatrixColumn(t, 2).length();
    return e[0] = s[0] * i, e[1] = s[1] * i, e[2] = s[2] * i, e[3] = 0, e[4] = s[4] * n, e[5] = s[5] * n, e[6] = s[6] * n, e[7] = 0, e[8] = s[8] * h, e[9] = s[9] * h, e[10] = s[10] * h, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, this;
  }
  makeRotationFromEuler(t) {
    const e = this.elements, s = t.x, i = t.y, n = t.z, h = Math.cos(s), a = Math.sin(s), r = Math.cos(i), o = Math.sin(i), l = Math.cos(n), u = Math.sin(n);
    if (t.order === "XYZ") {
      const d = h * l, c = h * u, p = a * l, x = a * u;
      e[0] = r * l, e[4] = -r * u, e[8] = o, e[1] = c + p * o, e[5] = d - x * o, e[9] = -a * r, e[2] = x - d * o, e[6] = p + c * o, e[10] = h * r;
    } else if (t.order === "YXZ") {
      const d = r * l, c = r * u, p = o * l, x = o * u;
      e[0] = d + x * a, e[4] = p * a - c, e[8] = h * o, e[1] = h * u, e[5] = h * l, e[9] = -a, e[2] = c * a - p, e[6] = x + d * a, e[10] = h * r;
    } else if (t.order === "ZXY") {
      const d = r * l, c = r * u, p = o * l, x = o * u;
      e[0] = d - x * a, e[4] = -h * u, e[8] = p + c * a, e[1] = c + p * a, e[5] = h * l, e[9] = x - d * a, e[2] = -h * o, e[6] = a, e[10] = h * r;
    } else if (t.order === "ZYX") {
      const d = h * l, c = h * u, p = a * l, x = a * u;
      e[0] = r * l, e[4] = p * o - c, e[8] = d * o + x, e[1] = r * u, e[5] = x * o + d, e[9] = c * o - p, e[2] = -o, e[6] = a * r, e[10] = h * r;
    } else if (t.order === "YZX") {
      const d = h * r, c = h * o, p = a * r, x = a * o;
      e[0] = r * l, e[4] = x - d * u, e[8] = p * u + c, e[1] = u, e[5] = h * l, e[9] = -a * l, e[2] = -o * l, e[6] = c * u + p, e[10] = d - x * u;
    } else if (t.order === "XZY") {
      const d = h * r, c = h * o, p = a * r, x = a * o;
      e[0] = r * l, e[4] = -u, e[8] = o * l, e[1] = d * u + x, e[5] = h * l, e[9] = c * u - p, e[2] = p * u - c, e[6] = a * l, e[10] = x * u + d;
    }
    return e[3] = 0, e[7] = 0, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, this;
  }
  makeRotationFromQuaternion(t) {
    return this.compose(Ee, t, ze);
  }
  lookAt(t, e, s) {
    const i = this.elements;
    return k.subVectors(t, e), k.lengthSq() === 0 && (k.z = 1), k.normalize(), L.crossVectors(s, k), L.lengthSq() === 0 && (Math.abs(s.z) === 1 ? k.x += 1e-4 : k.z += 1e-4, k.normalize(), L.crossVectors(s, k)), L.normalize(), mt.crossVectors(k, L), i[0] = L.x, i[4] = mt.x, i[8] = k.x, i[1] = L.y, i[5] = mt.y, i[9] = k.y, i[2] = L.z, i[6] = mt.z, i[10] = k.z, this;
  }
  multiply(t) {
    return this.multiplyMatrices(this, t);
  }
  premultiply(t) {
    return this.multiplyMatrices(t, this);
  }
  multiplyMatrices(t, e) {
    const s = t.elements, i = e.elements, n = this.elements, h = s[0], a = s[4], r = s[8], o = s[12], l = s[1], u = s[5], d = s[9], c = s[13], p = s[2], x = s[6], g = s[10], y = s[14], C = s[3], w = s[7], _ = s[11], M = s[15], F = i[0], f = i[4], D = i[8], G = i[12], $ = i[1], W = i[5], q = i[9], V = i[13], R = i[2], Y = i[6], nt = i[10], rt = i[14], at = i[3], ht = i[7], ot = i[11], lt = i[15];
    return n[0] = h * F + a * $ + r * R + o * at, n[4] = h * f + a * W + r * Y + o * ht, n[8] = h * D + a * q + r * nt + o * ot, n[12] = h * G + a * V + r * rt + o * lt, n[1] = l * F + u * $ + d * R + c * at, n[5] = l * f + u * W + d * Y + c * ht, n[9] = l * D + u * q + d * nt + c * ot, n[13] = l * G + u * V + d * rt + c * lt, n[2] = p * F + x * $ + g * R + y * at, n[6] = p * f + x * W + g * Y + y * ht, n[10] = p * D + x * q + g * nt + y * ot, n[14] = p * G + x * V + g * rt + y * lt, n[3] = C * F + w * $ + _ * R + M * at, n[7] = C * f + w * W + _ * Y + M * ht, n[11] = C * D + w * q + _ * nt + M * ot, n[15] = C * G + w * V + _ * rt + M * lt, this;
  }
  multiplyScalar(t) {
    const e = this.elements;
    return e[0] *= t, e[4] *= t, e[8] *= t, e[12] *= t, e[1] *= t, e[5] *= t, e[9] *= t, e[13] *= t, e[2] *= t, e[6] *= t, e[10] *= t, e[14] *= t, e[3] *= t, e[7] *= t, e[11] *= t, e[15] *= t, this;
  }
  determinant() {
    const t = this.elements, e = t[0], s = t[4], i = t[8], n = t[12], h = t[1], a = t[5], r = t[9], o = t[13], l = t[2], u = t[6], d = t[10], c = t[14], p = t[3], x = t[7], g = t[11], y = t[15];
    return p * (+n * r * u - i * o * u - n * a * d + s * o * d + i * a * c - s * r * c) + x * (+e * r * c - e * o * d + n * h * d - i * h * c + i * o * l - n * r * l) + g * (+e * o * u - e * a * c - n * h * u + s * h * c + n * a * l - s * o * l) + y * (-i * a * l - e * r * u + e * a * d + i * h * u - s * h * d + s * r * l);
  }
  transpose() {
    const t = this.elements;
    let e;
    return e = t[1], t[1] = t[4], t[4] = e, e = t[2], t[2] = t[8], t[8] = e, e = t[6], t[6] = t[9], t[9] = e, e = t[3], t[3] = t[12], t[12] = e, e = t[7], t[7] = t[13], t[13] = e, e = t[11], t[11] = t[14], t[14] = e, this;
  }
  setPosition(t, e, s) {
    const i = this.elements;
    return t.isVector3 ? (i[12] = t.x, i[13] = t.y, i[14] = t.z) : (i[12] = t, i[13] = e, i[14] = s), this;
  }
  invert() {
    const t = this.elements, e = t[0], s = t[1], i = t[2], n = t[3], h = t[4], a = t[5], r = t[6], o = t[7], l = t[8], u = t[9], d = t[10], c = t[11], p = t[12], x = t[13], g = t[14], y = t[15], C = u * g * o - x * d * o + x * r * c - a * g * c - u * r * y + a * d * y, w = p * d * o - l * g * o - p * r * c + h * g * c + l * r * y - h * d * y, _ = l * x * o - p * u * o + p * a * c - h * x * c - l * a * y + h * u * y, M = p * u * r - l * x * r - p * a * d + h * x * d + l * a * g - h * u * g, F = e * C + s * w + i * _ + n * M;
    if (F === 0)
      return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const f = 1 / F;
    return t[0] = C * f, t[1] = (x * d * n - u * g * n - x * i * c + s * g * c + u * i * y - s * d * y) * f, t[2] = (a * g * n - x * r * n + x * i * o - s * g * o - a * i * y + s * r * y) * f, t[3] = (u * r * n - a * d * n - u * i * o + s * d * o + a * i * c - s * r * c) * f, t[4] = w * f, t[5] = (l * g * n - p * d * n + p * i * c - e * g * c - l * i * y + e * d * y) * f, t[6] = (p * r * n - h * g * n - p * i * o + e * g * o + h * i * y - e * r * y) * f, t[7] = (h * d * n - l * r * n + l * i * o - e * d * o - h * i * c + e * r * c) * f, t[8] = _ * f, t[9] = (p * u * n - l * x * n - p * s * c + e * x * c + l * s * y - e * u * y) * f, t[10] = (h * x * n - p * a * n + p * s * o - e * x * o - h * s * y + e * a * y) * f, t[11] = (l * a * n - h * u * n - l * s * o + e * u * o + h * s * c - e * a * c) * f, t[12] = M * f, t[13] = (l * x * i - p * u * i + p * s * d - e * x * d - l * s * g + e * u * g) * f, t[14] = (p * a * i - h * x * i - p * s * r + e * x * r + h * s * g - e * a * g) * f, t[15] = (h * u * i - l * a * i + l * s * r - e * u * r - h * s * d + e * a * d) * f, this;
  }
  scale(t) {
    const e = this.elements, s = t.x, i = t.y, n = t.z;
    return e[0] *= s, e[4] *= i, e[8] *= n, e[1] *= s, e[5] *= i, e[9] *= n, e[2] *= s, e[6] *= i, e[10] *= n, e[3] *= s, e[7] *= i, e[11] *= n, this;
  }
  getMaxScaleOnAxis() {
    const t = this.elements, e = t[0] * t[0] + t[1] * t[1] + t[2] * t[2], s = t[4] * t[4] + t[5] * t[5] + t[6] * t[6], i = t[8] * t[8] + t[9] * t[9] + t[10] * t[10];
    return Math.sqrt(Math.max(e, s, i));
  }
  makeTranslation(t, e, s) {
    return t.isVector3 ? this.set(
      1,
      0,
      0,
      t.x,
      0,
      1,
      0,
      t.y,
      0,
      0,
      1,
      t.z,
      0,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      0,
      t,
      0,
      1,
      0,
      e,
      0,
      0,
      1,
      s,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationX(t) {
    const e = Math.cos(t), s = Math.sin(t);
    return this.set(
      1,
      0,
      0,
      0,
      0,
      e,
      -s,
      0,
      0,
      s,
      e,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationY(t) {
    const e = Math.cos(t), s = Math.sin(t);
    return this.set(
      e,
      0,
      s,
      0,
      0,
      1,
      0,
      0,
      -s,
      0,
      e,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationZ(t) {
    const e = Math.cos(t), s = Math.sin(t);
    return this.set(
      e,
      -s,
      0,
      0,
      s,
      e,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationAxis(t, e) {
    const s = Math.cos(e), i = Math.sin(e), n = 1 - s, h = t.x, a = t.y, r = t.z, o = n * h, l = n * a;
    return this.set(
      o * h + s,
      o * a - i * r,
      o * r + i * a,
      0,
      o * a + i * r,
      l * a + s,
      l * r - i * h,
      0,
      o * r - i * a,
      l * r + i * h,
      n * r * r + s,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeScale(t, e, s) {
    return this.set(
      t,
      0,
      0,
      0,
      0,
      e,
      0,
      0,
      0,
      0,
      s,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeShear(t, e, s, i, n, h) {
    return this.set(
      1,
      s,
      n,
      0,
      t,
      1,
      h,
      0,
      e,
      i,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  compose(t, e, s) {
    const i = this.elements, n = e._x, h = e._y, a = e._z, r = e._w, o = n + n, l = h + h, u = a + a, d = n * o, c = n * l, p = n * u, x = h * l, g = h * u, y = a * u, C = r * o, w = r * l, _ = r * u, M = s.x, F = s.y, f = s.z;
    return i[0] = (1 - (x + y)) * M, i[1] = (c + _) * M, i[2] = (p - w) * M, i[3] = 0, i[4] = (c - _) * F, i[5] = (1 - (d + y)) * F, i[6] = (g + C) * F, i[7] = 0, i[8] = (p + w) * f, i[9] = (g - C) * f, i[10] = (1 - (d + x)) * f, i[11] = 0, i[12] = t.x, i[13] = t.y, i[14] = t.z, i[15] = 1, this;
  }
  decompose(t, e, s) {
    const i = this.elements;
    let n = Z.set(i[0], i[1], i[2]).length();
    const h = Z.set(i[4], i[5], i[6]).length(), a = Z.set(i[8], i[9], i[10]).length();
    this.determinant() < 0 && (n = -n), t.x = i[12], t.y = i[13], t.z = i[14], N.copy(this);
    const o = 1 / n, l = 1 / h, u = 1 / a;
    return N.elements[0] *= o, N.elements[1] *= o, N.elements[2] *= o, N.elements[4] *= l, N.elements[5] *= l, N.elements[6] *= l, N.elements[8] *= u, N.elements[9] *= u, N.elements[10] *= u, e.setFromRotationMatrix(N), s.x = n, s.y = h, s.z = a, this;
  }
  makePerspective(t, e, s, i, n, h, a = ct) {
    const r = this.elements, o = 2 * n / (e - t), l = 2 * n / (s - i), u = (e + t) / (e - t), d = (s + i) / (s - i);
    let c, p;
    if (a === ct)
      c = -(h + n) / (h - n), p = -2 * h * n / (h - n);
    else if (a === Gt)
      c = -h / (h - n), p = -h * n / (h - n);
    else
      throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: " + a);
    return r[0] = o, r[4] = 0, r[8] = u, r[12] = 0, r[1] = 0, r[5] = l, r[9] = d, r[13] = 0, r[2] = 0, r[6] = 0, r[10] = c, r[14] = p, r[3] = 0, r[7] = 0, r[11] = -1, r[15] = 0, this;
  }
  makeOrthographic(t, e, s, i, n, h, a = ct) {
    const r = this.elements, o = 1 / (e - t), l = 1 / (s - i), u = 1 / (h - n), d = (e + t) * o, c = (s + i) * l;
    let p, x;
    if (a === ct)
      p = (h + n) * u, x = -2 * u;
    else if (a === Gt)
      p = n * u, x = -1 * u;
    else
      throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: " + a);
    return r[0] = 2 * o, r[4] = 0, r[8] = 0, r[12] = -d, r[1] = 0, r[5] = 2 * l, r[9] = 0, r[13] = -c, r[2] = 0, r[6] = 0, r[10] = x, r[14] = -p, r[3] = 0, r[7] = 0, r[11] = 0, r[15] = 1, this;
  }
  equals(t) {
    const e = this.elements, s = t.elements;
    for (let i = 0; i < 16; i++)
      if (e[i] !== s[i])
        return !1;
    return !0;
  }
  fromArray(t, e = 0) {
    for (let s = 0; s < 16; s++)
      this.elements[s] = t[s + e];
    return this;
  }
  toArray(t = [], e = 0) {
    const s = this.elements;
    return t[e] = s[0], t[e + 1] = s[1], t[e + 2] = s[2], t[e + 3] = s[3], t[e + 4] = s[4], t[e + 5] = s[5], t[e + 6] = s[6], t[e + 7] = s[7], t[e + 8] = s[8], t[e + 9] = s[9], t[e + 10] = s[10], t[e + 11] = s[11], t[e + 12] = s[12], t[e + 13] = s[13], t[e + 14] = s[14], t[e + 15] = s[15], t;
  }
}
const Z = /* @__PURE__ */ new E(), N = /* @__PURE__ */ new B(), Ee = /* @__PURE__ */ new E(0, 0, 0), ze = /* @__PURE__ */ new E(1, 1, 1), L = /* @__PURE__ */ new E(), mt = /* @__PURE__ */ new E(), k = /* @__PURE__ */ new E(), $t = /* @__PURE__ */ new B(), Vt = /* @__PURE__ */ new it();
class yt {
  constructor(t = 0, e = 0, s = 0, i = yt.DEFAULT_ORDER) {
    this.isEuler = !0, this._x = t, this._y = e, this._z = s, this._order = i;
  }
  get x() {
    return this._x;
  }
  set x(t) {
    this._x = t, this._onChangeCallback();
  }
  get y() {
    return this._y;
  }
  set y(t) {
    this._y = t, this._onChangeCallback();
  }
  get z() {
    return this._z;
  }
  set z(t) {
    this._z = t, this._onChangeCallback();
  }
  get order() {
    return this._order;
  }
  set order(t) {
    this._order = t, this._onChangeCallback();
  }
  set(t, e, s, i = this._order) {
    return this._x = t, this._y = e, this._z = s, this._order = i, this._onChangeCallback(), this;
  }
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._order);
  }
  copy(t) {
    return this._x = t._x, this._y = t._y, this._z = t._z, this._order = t._order, this._onChangeCallback(), this;
  }
  setFromRotationMatrix(t, e = this._order, s = !0) {
    const i = t.elements, n = i[0], h = i[4], a = i[8], r = i[1], o = i[5], l = i[9], u = i[2], d = i[6], c = i[10];
    switch (e) {
      case "XYZ":
        this._y = Math.asin(S(a, -1, 1)), Math.abs(a) < 0.9999999 ? (this._x = Math.atan2(-l, c), this._z = Math.atan2(-h, n)) : (this._x = Math.atan2(d, o), this._z = 0);
        break;
      case "YXZ":
        this._x = Math.asin(-S(l, -1, 1)), Math.abs(l) < 0.9999999 ? (this._y = Math.atan2(a, c), this._z = Math.atan2(r, o)) : (this._y = Math.atan2(-u, n), this._z = 0);
        break;
      case "ZXY":
        this._x = Math.asin(S(d, -1, 1)), Math.abs(d) < 0.9999999 ? (this._y = Math.atan2(-u, c), this._z = Math.atan2(-h, o)) : (this._y = 0, this._z = Math.atan2(r, n));
        break;
      case "ZYX":
        this._y = Math.asin(-S(u, -1, 1)), Math.abs(u) < 0.9999999 ? (this._x = Math.atan2(d, c), this._z = Math.atan2(r, n)) : (this._x = 0, this._z = Math.atan2(-h, o));
        break;
      case "YZX":
        this._z = Math.asin(S(r, -1, 1)), Math.abs(r) < 0.9999999 ? (this._x = Math.atan2(-l, o), this._y = Math.atan2(-u, n)) : (this._x = 0, this._y = Math.atan2(a, c));
        break;
      case "XZY":
        this._z = Math.asin(-S(h, -1, 1)), Math.abs(h) < 0.9999999 ? (this._x = Math.atan2(d, o), this._y = Math.atan2(a, n)) : (this._x = Math.atan2(-l, c), this._y = 0);
        break;
      default:
        console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: " + e);
    }
    return this._order = e, s === !0 && this._onChangeCallback(), this;
  }
  setFromQuaternion(t, e, s) {
    return $t.makeRotationFromQuaternion(t), this.setFromRotationMatrix($t, e, s);
  }
  setFromVector3(t, e = this._order) {
    return this.set(t.x, t.y, t.z, e);
  }
  reorder(t) {
    return Vt.setFromEuler(this), this.setFromQuaternion(Vt, t);
  }
  equals(t) {
    return t._x === this._x && t._y === this._y && t._z === this._z && t._order === this._order;
  }
  fromArray(t) {
    return this._x = t[0], this._y = t[1], this._z = t[2], t[3] !== void 0 && (this._order = t[3]), this._onChangeCallback(), this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this._x, t[e + 1] = this._y, t[e + 2] = this._z, t[e + 3] = this._order, t;
  }
  _onChange(t) {
    return this._onChangeCallback = t, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._order;
  }
}
yt.DEFAULT_ORDER = "XYZ";
class Ae {
  constructor() {
    this.mask = 1;
  }
  set(t) {
    this.mask = (1 << t | 0) >>> 0;
  }
  enable(t) {
    this.mask |= 1 << t | 0;
  }
  enableAll() {
    this.mask = -1;
  }
  toggle(t) {
    this.mask ^= 1 << t | 0;
  }
  disable(t) {
    this.mask &= ~(1 << t | 0);
  }
  disableAll() {
    this.mask = 0;
  }
  test(t) {
    return (this.mask & t.mask) !== 0;
  }
  isEnabled(t) {
    return (this.mask & (1 << t | 0)) !== 0;
  }
}
let Se = 0;
const Yt = /* @__PURE__ */ new E(), J = /* @__PURE__ */ new it(), O = /* @__PURE__ */ new B(), pt = /* @__PURE__ */ new E(), tt = /* @__PURE__ */ new E(), ke = /* @__PURE__ */ new E(), Ie = /* @__PURE__ */ new it(), Xt = /* @__PURE__ */ new E(1, 0, 0), Zt = /* @__PURE__ */ new E(0, 1, 0), Jt = /* @__PURE__ */ new E(0, 0, 1), Qt = { type: "added" }, Ne = { type: "removed" }, Q = { type: "childadded", child: null }, Ct = { type: "childremoved", child: null };
class H extends ne {
  constructor() {
    super(), this.isObject3D = !0, Object.defineProperty(this, "id", { value: Se++ }), this.uuid = St(), this.name = "", this.type = "Object3D", this.parent = null, this.children = [], this.up = H.DEFAULT_UP.clone();
    const t = new E(), e = new yt(), s = new it(), i = new E(1, 1, 1);
    function n() {
      s.setFromEuler(e, !1);
    }
    function h() {
      e.setFromQuaternion(s, void 0, !1);
    }
    e._onChange(n), s._onChange(h), Object.defineProperties(this, {
      position: {
        configurable: !0,
        enumerable: !0,
        value: t
      },
      rotation: {
        configurable: !0,
        enumerable: !0,
        value: e
      },
      quaternion: {
        configurable: !0,
        enumerable: !0,
        value: s
      },
      scale: {
        configurable: !0,
        enumerable: !0,
        value: i
      },
      modelViewMatrix: {
        value: new B()
      },
      normalMatrix: {
        value: new j()
      }
    }), this.matrix = new B(), this.matrixWorld = new B(), this.matrixAutoUpdate = H.DEFAULT_MATRIX_AUTO_UPDATE, this.matrixWorldAutoUpdate = H.DEFAULT_MATRIX_WORLD_AUTO_UPDATE, this.matrixWorldNeedsUpdate = !1, this.layers = new Ae(), this.visible = !0, this.castShadow = !1, this.receiveShadow = !1, this.frustumCulled = !0, this.renderOrder = 0, this.animations = [], this.userData = {};
  }
  onBeforeShadow() {
  }
  onAfterShadow() {
  }
  onBeforeRender() {
  }
  onAfterRender() {
  }
  applyMatrix4(t) {
    this.matrixAutoUpdate && this.updateMatrix(), this.matrix.premultiply(t), this.matrix.decompose(this.position, this.quaternion, this.scale);
  }
  applyQuaternion(t) {
    return this.quaternion.premultiply(t), this;
  }
  setRotationFromAxisAngle(t, e) {
    this.quaternion.setFromAxisAngle(t, e);
  }
  setRotationFromEuler(t) {
    this.quaternion.setFromEuler(t, !0);
  }
  setRotationFromMatrix(t) {
    this.quaternion.setFromRotationMatrix(t);
  }
  setRotationFromQuaternion(t) {
    this.quaternion.copy(t);
  }
  rotateOnAxis(t, e) {
    return J.setFromAxisAngle(t, e), this.quaternion.multiply(J), this;
  }
  rotateOnWorldAxis(t, e) {
    return J.setFromAxisAngle(t, e), this.quaternion.premultiply(J), this;
  }
  rotateX(t) {
    return this.rotateOnAxis(Xt, t);
  }
  rotateY(t) {
    return this.rotateOnAxis(Zt, t);
  }
  rotateZ(t) {
    return this.rotateOnAxis(Jt, t);
  }
  translateOnAxis(t, e) {
    return Yt.copy(t).applyQuaternion(this.quaternion), this.position.add(Yt.multiplyScalar(e)), this;
  }
  translateX(t) {
    return this.translateOnAxis(Xt, t);
  }
  translateY(t) {
    return this.translateOnAxis(Zt, t);
  }
  translateZ(t) {
    return this.translateOnAxis(Jt, t);
  }
  localToWorld(t) {
    return this.updateWorldMatrix(!0, !1), t.applyMatrix4(this.matrixWorld);
  }
  worldToLocal(t) {
    return this.updateWorldMatrix(!0, !1), t.applyMatrix4(O.copy(this.matrixWorld).invert());
  }
  lookAt(t, e, s) {
    t.isVector3 ? pt.copy(t) : pt.set(t, e, s);
    const i = this.parent;
    this.updateWorldMatrix(!0, !1), tt.setFromMatrixPosition(this.matrixWorld), this.isCamera || this.isLight ? O.lookAt(tt, pt, this.up) : O.lookAt(pt, tt, this.up), this.quaternion.setFromRotationMatrix(O), i && (O.extractRotation(i.matrixWorld), J.setFromRotationMatrix(O), this.quaternion.premultiply(J.invert()));
  }
  add(t) {
    if (arguments.length > 1) {
      for (let e = 0; e < arguments.length; e++)
        this.add(arguments[e]);
      return this;
    }
    return t === this ? (console.error("THREE.Object3D.add: object can't be added as a child of itself.", t), this) : (t && t.isObject3D ? (t.removeFromParent(), t.parent = this, this.children.push(t), t.dispatchEvent(Qt), Q.child = t, this.dispatchEvent(Q), Q.child = null) : console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.", t), this);
  }
  remove(t) {
    if (arguments.length > 1) {
      for (let s = 0; s < arguments.length; s++)
        this.remove(arguments[s]);
      return this;
    }
    const e = this.children.indexOf(t);
    return e !== -1 && (t.parent = null, this.children.splice(e, 1), t.dispatchEvent(Ne), Ct.child = t, this.dispatchEvent(Ct), Ct.child = null), this;
  }
  removeFromParent() {
    const t = this.parent;
    return t !== null && t.remove(this), this;
  }
  clear() {
    return this.remove(...this.children);
  }
  attach(t) {
    return this.updateWorldMatrix(!0, !1), O.copy(this.matrixWorld).invert(), t.parent !== null && (t.parent.updateWorldMatrix(!0, !1), O.multiply(t.parent.matrixWorld)), t.applyMatrix4(O), t.removeFromParent(), t.parent = this, this.children.push(t), t.updateWorldMatrix(!1, !0), t.dispatchEvent(Qt), Q.child = t, this.dispatchEvent(Q), Q.child = null, this;
  }
  getObjectById(t) {
    return this.getObjectByProperty("id", t);
  }
  getObjectByName(t) {
    return this.getObjectByProperty("name", t);
  }
  getObjectByProperty(t, e) {
    if (this[t] === e)
      return this;
    for (let s = 0, i = this.children.length; s < i; s++) {
      const h = this.children[s].getObjectByProperty(t, e);
      if (h !== void 0)
        return h;
    }
  }
  getObjectsByProperty(t, e, s = []) {
    this[t] === e && s.push(this);
    const i = this.children;
    for (let n = 0, h = i.length; n < h; n++)
      i[n].getObjectsByProperty(t, e, s);
    return s;
  }
  getWorldPosition(t) {
    return this.updateWorldMatrix(!0, !1), t.setFromMatrixPosition(this.matrixWorld);
  }
  getWorldQuaternion(t) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(tt, t, ke), t;
  }
  getWorldScale(t) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(tt, Ie, t), t;
  }
  getWorldDirection(t) {
    this.updateWorldMatrix(!0, !1);
    const e = this.matrixWorld.elements;
    return t.set(e[8], e[9], e[10]).normalize();
  }
  raycast() {
  }
  traverse(t) {
    t(this);
    const e = this.children;
    for (let s = 0, i = e.length; s < i; s++)
      e[s].traverse(t);
  }
  traverseVisible(t) {
    if (this.visible === !1)
      return;
    t(this);
    const e = this.children;
    for (let s = 0, i = e.length; s < i; s++)
      e[s].traverseVisible(t);
  }
  traverseAncestors(t) {
    const e = this.parent;
    e !== null && (t(e), e.traverseAncestors(t));
  }
  updateMatrix() {
    this.matrix.compose(this.position, this.quaternion, this.scale), this.matrixWorldNeedsUpdate = !0;
  }
  updateMatrixWorld(t) {
    this.matrixAutoUpdate && this.updateMatrix(), (this.matrixWorldNeedsUpdate || t) && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix), this.matrixWorldNeedsUpdate = !1, t = !0);
    const e = this.children;
    for (let s = 0, i = e.length; s < i; s++) {
      const n = e[s];
      (n.matrixWorldAutoUpdate === !0 || t === !0) && n.updateMatrixWorld(t);
    }
  }
  updateWorldMatrix(t, e) {
    const s = this.parent;
    if (t === !0 && s !== null && s.matrixWorldAutoUpdate === !0 && s.updateWorldMatrix(!0, !1), this.matrixAutoUpdate && this.updateMatrix(), this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix), e === !0) {
      const i = this.children;
      for (let n = 0, h = i.length; n < h; n++) {
        const a = i[n];
        a.matrixWorldAutoUpdate === !0 && a.updateWorldMatrix(!1, !0);
      }
    }
  }
  toJSON(t) {
    const e = t === void 0 || typeof t == "string", s = {};
    e && (t = {
      geometries: {},
      materials: {},
      textures: {},
      images: {},
      shapes: {},
      skeletons: {},
      animations: {},
      nodes: {}
    }, s.metadata = {
      version: 4.6,
      type: "Object",
      generator: "Object3D.toJSON"
    });
    const i = {};
    i.uuid = this.uuid, i.type = this.type, this.name !== "" && (i.name = this.name), this.castShadow === !0 && (i.castShadow = !0), this.receiveShadow === !0 && (i.receiveShadow = !0), this.visible === !1 && (i.visible = !1), this.frustumCulled === !1 && (i.frustumCulled = !1), this.renderOrder !== 0 && (i.renderOrder = this.renderOrder), Object.keys(this.userData).length > 0 && (i.userData = this.userData), i.layers = this.layers.mask, i.matrix = this.matrix.toArray(), i.up = this.up.toArray(), this.matrixAutoUpdate === !1 && (i.matrixAutoUpdate = !1), this.isInstancedMesh && (i.type = "InstancedMesh", i.count = this.count, i.instanceMatrix = this.instanceMatrix.toJSON(), this.instanceColor !== null && (i.instanceColor = this.instanceColor.toJSON())), this.isBatchedMesh && (i.type = "BatchedMesh", i.perObjectFrustumCulled = this.perObjectFrustumCulled, i.sortObjects = this.sortObjects, i.drawRanges = this._drawRanges, i.reservedRanges = this._reservedRanges, i.visibility = this._visibility, i.active = this._active, i.bounds = this._bounds.map((a) => ({
      boxInitialized: a.boxInitialized,
      boxMin: a.box.min.toArray(),
      boxMax: a.box.max.toArray(),
      sphereInitialized: a.sphereInitialized,
      sphereRadius: a.sphere.radius,
      sphereCenter: a.sphere.center.toArray()
    })), i.maxGeometryCount = this._maxGeometryCount, i.maxVertexCount = this._maxVertexCount, i.maxIndexCount = this._maxIndexCount, i.geometryInitialized = this._geometryInitialized, i.geometryCount = this._geometryCount, i.matricesTexture = this._matricesTexture.toJSON(t), this.boundingSphere !== null && (i.boundingSphere = {
      center: i.boundingSphere.center.toArray(),
      radius: i.boundingSphere.radius
    }), this.boundingBox !== null && (i.boundingBox = {
      min: i.boundingBox.min.toArray(),
      max: i.boundingBox.max.toArray()
    }));
    function n(a, r) {
      return a[r.uuid] === void 0 && (a[r.uuid] = r.toJSON(t)), r.uuid;
    }
    if (this.isScene)
      this.background && (this.background.isColor ? i.background = this.background.toJSON() : this.background.isTexture && (i.background = this.background.toJSON(t).uuid)), this.environment && this.environment.isTexture && this.environment.isRenderTargetTexture !== !0 && (i.environment = this.environment.toJSON(t).uuid);
    else if (this.isMesh || this.isLine || this.isPoints) {
      i.geometry = n(t.geometries, this.geometry);
      const a = this.geometry.parameters;
      if (a !== void 0 && a.shapes !== void 0) {
        const r = a.shapes;
        if (Array.isArray(r))
          for (let o = 0, l = r.length; o < l; o++) {
            const u = r[o];
            n(t.shapes, u);
          }
        else
          n(t.shapes, r);
      }
    }
    if (this.isSkinnedMesh && (i.bindMode = this.bindMode, i.bindMatrix = this.bindMatrix.toArray(), this.skeleton !== void 0 && (n(t.skeletons, this.skeleton), i.skeleton = this.skeleton.uuid)), this.material !== void 0)
      if (Array.isArray(this.material)) {
        const a = [];
        for (let r = 0, o = this.material.length; r < o; r++)
          a.push(n(t.materials, this.material[r]));
        i.material = a;
      } else
        i.material = n(t.materials, this.material);
    if (this.children.length > 0) {
      i.children = [];
      for (let a = 0; a < this.children.length; a++)
        i.children.push(this.children[a].toJSON(t).object);
    }
    if (this.animations.length > 0) {
      i.animations = [];
      for (let a = 0; a < this.animations.length; a++) {
        const r = this.animations[a];
        i.animations.push(n(t.animations, r));
      }
    }
    if (e) {
      const a = h(t.geometries), r = h(t.materials), o = h(t.textures), l = h(t.images), u = h(t.shapes), d = h(t.skeletons), c = h(t.animations), p = h(t.nodes);
      a.length > 0 && (s.geometries = a), r.length > 0 && (s.materials = r), o.length > 0 && (s.textures = o), l.length > 0 && (s.images = l), u.length > 0 && (s.shapes = u), d.length > 0 && (s.skeletons = d), c.length > 0 && (s.animations = c), p.length > 0 && (s.nodes = p);
    }
    return s.object = i, s;
    function h(a) {
      const r = [];
      for (const o in a) {
        const l = a[o];
        delete l.metadata, r.push(l);
      }
      return r;
    }
  }
  clone(t) {
    return new this.constructor().copy(this, t);
  }
  copy(t, e = !0) {
    if (this.name = t.name, this.up.copy(t.up), this.position.copy(t.position), this.rotation.order = t.rotation.order, this.quaternion.copy(t.quaternion), this.scale.copy(t.scale), this.matrix.copy(t.matrix), this.matrixWorld.copy(t.matrixWorld), this.matrixAutoUpdate = t.matrixAutoUpdate, this.matrixWorldAutoUpdate = t.matrixWorldAutoUpdate, this.matrixWorldNeedsUpdate = t.matrixWorldNeedsUpdate, this.layers.mask = t.layers.mask, this.visible = t.visible, this.castShadow = t.castShadow, this.receiveShadow = t.receiveShadow, this.frustumCulled = t.frustumCulled, this.renderOrder = t.renderOrder, this.animations = t.animations.slice(), this.userData = JSON.parse(JSON.stringify(t.userData)), e === !0)
      for (let s = 0; s < t.children.length; s++) {
        const i = t.children[s];
        this.add(i.clone());
      }
    return this;
  }
}
H.DEFAULT_UP = /* @__PURE__ */ new E(0, 1, 0);
H.DEFAULT_MATRIX_AUTO_UPDATE = !0;
H.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = !0;
const re = {
  aliceblue: 15792383,
  antiquewhite: 16444375,
  aqua: 65535,
  aquamarine: 8388564,
  azure: 15794175,
  beige: 16119260,
  bisque: 16770244,
  black: 0,
  blanchedalmond: 16772045,
  blue: 255,
  blueviolet: 9055202,
  brown: 10824234,
  burlywood: 14596231,
  cadetblue: 6266528,
  chartreuse: 8388352,
  chocolate: 13789470,
  coral: 16744272,
  cornflowerblue: 6591981,
  cornsilk: 16775388,
  crimson: 14423100,
  cyan: 65535,
  darkblue: 139,
  darkcyan: 35723,
  darkgoldenrod: 12092939,
  darkgray: 11119017,
  darkgreen: 25600,
  darkgrey: 11119017,
  darkkhaki: 12433259,
  darkmagenta: 9109643,
  darkolivegreen: 5597999,
  darkorange: 16747520,
  darkorchid: 10040012,
  darkred: 9109504,
  darksalmon: 15308410,
  darkseagreen: 9419919,
  darkslateblue: 4734347,
  darkslategray: 3100495,
  darkslategrey: 3100495,
  darkturquoise: 52945,
  darkviolet: 9699539,
  deeppink: 16716947,
  deepskyblue: 49151,
  dimgray: 6908265,
  dimgrey: 6908265,
  dodgerblue: 2003199,
  firebrick: 11674146,
  floralwhite: 16775920,
  forestgreen: 2263842,
  fuchsia: 16711935,
  gainsboro: 14474460,
  ghostwhite: 16316671,
  gold: 16766720,
  goldenrod: 14329120,
  gray: 8421504,
  green: 32768,
  greenyellow: 11403055,
  grey: 8421504,
  honeydew: 15794160,
  hotpink: 16738740,
  indianred: 13458524,
  indigo: 4915330,
  ivory: 16777200,
  khaki: 15787660,
  lavender: 15132410,
  lavenderblush: 16773365,
  lawngreen: 8190976,
  lemonchiffon: 16775885,
  lightblue: 11393254,
  lightcoral: 15761536,
  lightcyan: 14745599,
  lightgoldenrodyellow: 16448210,
  lightgray: 13882323,
  lightgreen: 9498256,
  lightgrey: 13882323,
  lightpink: 16758465,
  lightsalmon: 16752762,
  lightseagreen: 2142890,
  lightskyblue: 8900346,
  lightslategray: 7833753,
  lightslategrey: 7833753,
  lightsteelblue: 11584734,
  lightyellow: 16777184,
  lime: 65280,
  limegreen: 3329330,
  linen: 16445670,
  magenta: 16711935,
  maroon: 8388608,
  mediumaquamarine: 6737322,
  mediumblue: 205,
  mediumorchid: 12211667,
  mediumpurple: 9662683,
  mediumseagreen: 3978097,
  mediumslateblue: 8087790,
  mediumspringgreen: 64154,
  mediumturquoise: 4772300,
  mediumvioletred: 13047173,
  midnightblue: 1644912,
  mintcream: 16121850,
  mistyrose: 16770273,
  moccasin: 16770229,
  navajowhite: 16768685,
  navy: 128,
  oldlace: 16643558,
  olive: 8421376,
  olivedrab: 7048739,
  orange: 16753920,
  orangered: 16729344,
  orchid: 14315734,
  palegoldenrod: 15657130,
  palegreen: 10025880,
  paleturquoise: 11529966,
  palevioletred: 14381203,
  papayawhip: 16773077,
  peachpuff: 16767673,
  peru: 13468991,
  pink: 16761035,
  plum: 14524637,
  powderblue: 11591910,
  purple: 8388736,
  rebeccapurple: 6697881,
  red: 16711680,
  rosybrown: 12357519,
  royalblue: 4286945,
  saddlebrown: 9127187,
  salmon: 16416882,
  sandybrown: 16032864,
  seagreen: 3050327,
  seashell: 16774638,
  sienna: 10506797,
  silver: 12632256,
  skyblue: 8900331,
  slateblue: 6970061,
  slategray: 7372944,
  slategrey: 7372944,
  snow: 16775930,
  springgreen: 65407,
  steelblue: 4620980,
  tan: 13808780,
  teal: 32896,
  thistle: 14204888,
  tomato: 16737095,
  turquoise: 4251856,
  violet: 15631086,
  wheat: 16113331,
  white: 16777215,
  whitesmoke: 16119285,
  yellow: 16776960,
  yellowgreen: 10145074
}, v = { h: 0, s: 0, l: 0 }, xt = { h: 0, s: 0, l: 0 };
function Et(m, t, e) {
  return e < 0 && (e += 1), e > 1 && (e -= 1), e < 1 / 6 ? m + (t - m) * 6 * e : e < 1 / 2 ? t : e < 2 / 3 ? m + (t - m) * 6 * (2 / 3 - e) : m;
}
class kt {
  constructor(t, e, s) {
    return this.isColor = !0, this.r = 1, this.g = 1, this.b = 1, this.set(t, e, s);
  }
  set(t, e, s) {
    if (e === void 0 && s === void 0) {
      const i = t;
      i && i.isColor ? this.copy(i) : typeof i == "number" ? this.setHex(i) : typeof i == "string" && this.setStyle(i);
    } else
      this.setRGB(t, e, s);
    return this;
  }
  setScalar(t) {
    return this.r = t, this.g = t, this.b = t, this;
  }
  setHex(t, e = P) {
    return t = Math.floor(t), this.r = (t >> 16 & 255) / 255, this.g = (t >> 8 & 255) / 255, this.b = (t & 255) / 255, I.toWorkingColorSpace(this, e), this;
  }
  setRGB(t, e, s, i = I.workingColorSpace) {
    return this.r = t, this.g = e, this.b = s, I.toWorkingColorSpace(this, i), this;
  }
  setHSL(t, e, s, i = I.workingColorSpace) {
    if (t = Me(t, 1), e = S(e, 0, 1), s = S(s, 0, 1), e === 0)
      this.r = this.g = this.b = s;
    else {
      const n = s <= 0.5 ? s * (1 + e) : s + e - s * e, h = 2 * s - n;
      this.r = Et(h, n, t + 1 / 3), this.g = Et(h, n, t), this.b = Et(h, n, t - 1 / 3);
    }
    return I.toWorkingColorSpace(this, i), this;
  }
  setStyle(t, e = P) {
    function s(n) {
      n !== void 0 && parseFloat(n) < 1 && console.warn("THREE.Color: Alpha component of " + t + " will be ignored.");
    }
    let i;
    if (i = /^(\w+)\(([^\)]*)\)/.exec(t)) {
      let n;
      const h = i[1], a = i[2];
      switch (h) {
        case "rgb":
        case "rgba":
          if (n = /^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return s(n[4]), this.setRGB(
              Math.min(255, parseInt(n[1], 10)) / 255,
              Math.min(255, parseInt(n[2], 10)) / 255,
              Math.min(255, parseInt(n[3], 10)) / 255,
              e
            );
          if (n = /^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return s(n[4]), this.setRGB(
              Math.min(100, parseInt(n[1], 10)) / 100,
              Math.min(100, parseInt(n[2], 10)) / 100,
              Math.min(100, parseInt(n[3], 10)) / 100,
              e
            );
          break;
        case "hsl":
        case "hsla":
          if (n = /^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))
            return s(n[4]), this.setHSL(
              parseFloat(n[1]) / 360,
              parseFloat(n[2]) / 100,
              parseFloat(n[3]) / 100,
              e
            );
          break;
        default:
          console.warn("THREE.Color: Unknown color model " + t);
      }
    } else if (i = /^\#([A-Fa-f\d]+)$/.exec(t)) {
      const n = i[1], h = n.length;
      if (h === 3)
        return this.setRGB(
          parseInt(n.charAt(0), 16) / 15,
          parseInt(n.charAt(1), 16) / 15,
          parseInt(n.charAt(2), 16) / 15,
          e
        );
      if (h === 6)
        return this.setHex(parseInt(n, 16), e);
      console.warn("THREE.Color: Invalid hex color " + t);
    } else if (t && t.length > 0)
      return this.setColorName(t, e);
    return this;
  }
  setColorName(t, e = P) {
    const s = re[t.toLowerCase()];
    return s !== void 0 ? this.setHex(s, e) : console.warn("THREE.Color: Unknown color " + t), this;
  }
  clone() {
    return new this.constructor(this.r, this.g, this.b);
  }
  copy(t) {
    return this.r = t.r, this.g = t.g, this.b = t.b, this;
  }
  copySRGBToLinear(t) {
    return this.r = K(t.r), this.g = K(t.g), this.b = K(t.b), this;
  }
  copyLinearToSRGB(t) {
    return this.r = _t(t.r), this.g = _t(t.g), this.b = _t(t.b), this;
  }
  convertSRGBToLinear() {
    return this.copySRGBToLinear(this), this;
  }
  convertLinearToSRGB() {
    return this.copyLinearToSRGB(this), this;
  }
  getHex(t = P) {
    return I.fromWorkingColorSpace(A.copy(this), t), Math.round(S(A.r * 255, 0, 255)) * 65536 + Math.round(S(A.g * 255, 0, 255)) * 256 + Math.round(S(A.b * 255, 0, 255));
  }
  getHexString(t = P) {
    return ("000000" + this.getHex(t).toString(16)).slice(-6);
  }
  getHSL(t, e = I.workingColorSpace) {
    I.fromWorkingColorSpace(A.copy(this), e);
    const s = A.r, i = A.g, n = A.b, h = Math.max(s, i, n), a = Math.min(s, i, n);
    let r, o;
    const l = (a + h) / 2;
    if (a === h)
      r = 0, o = 0;
    else {
      const u = h - a;
      switch (o = l <= 0.5 ? u / (h + a) : u / (2 - h - a), h) {
        case s:
          r = (i - n) / u + (i < n ? 6 : 0);
          break;
        case i:
          r = (n - s) / u + 2;
          break;
        case n:
          r = (s - i) / u + 4;
          break;
      }
      r /= 6;
    }
    return t.h = r, t.s = o, t.l = l, t;
  }
  getRGB(t, e = I.workingColorSpace) {
    return I.fromWorkingColorSpace(A.copy(this), e), t.r = A.r, t.g = A.g, t.b = A.b, t;
  }
  getStyle(t = P) {
    I.fromWorkingColorSpace(A.copy(this), t);
    const e = A.r, s = A.g, i = A.b;
    return t !== P ? `color(${t} ${e.toFixed(3)} ${s.toFixed(3)} ${i.toFixed(3)})` : `rgb(${Math.round(e * 255)},${Math.round(s * 255)},${Math.round(i * 255)})`;
  }
  offsetHSL(t, e, s) {
    return this.getHSL(v), this.setHSL(v.h + t, v.s + e, v.l + s);
  }
  add(t) {
    return this.r += t.r, this.g += t.g, this.b += t.b, this;
  }
  addColors(t, e) {
    return this.r = t.r + e.r, this.g = t.g + e.g, this.b = t.b + e.b, this;
  }
  addScalar(t) {
    return this.r += t, this.g += t, this.b += t, this;
  }
  sub(t) {
    return this.r = Math.max(0, this.r - t.r), this.g = Math.max(0, this.g - t.g), this.b = Math.max(0, this.b - t.b), this;
  }
  multiply(t) {
    return this.r *= t.r, this.g *= t.g, this.b *= t.b, this;
  }
  multiplyScalar(t) {
    return this.r *= t, this.g *= t, this.b *= t, this;
  }
  lerp(t, e) {
    return this.r += (t.r - this.r) * e, this.g += (t.g - this.g) * e, this.b += (t.b - this.b) * e, this;
  }
  lerpColors(t, e, s) {
    return this.r = t.r + (e.r - t.r) * s, this.g = t.g + (e.g - t.g) * s, this.b = t.b + (e.b - t.b) * s, this;
  }
  lerpHSL(t, e) {
    this.getHSL(v), t.getHSL(xt);
    const s = Mt(v.h, xt.h, e), i = Mt(v.s, xt.s, e), n = Mt(v.l, xt.l, e);
    return this.setHSL(s, i, n), this;
  }
  setFromVector3(t) {
    return this.r = t.x, this.g = t.y, this.b = t.z, this;
  }
  applyMatrix3(t) {
    const e = this.r, s = this.g, i = this.b, n = t.elements;
    return this.r = n[0] * e + n[3] * s + n[6] * i, this.g = n[1] * e + n[4] * s + n[7] * i, this.b = n[2] * e + n[5] * s + n[8] * i, this;
  }
  equals(t) {
    return t.r === this.r && t.g === this.g && t.b === this.b;
  }
  fromArray(t, e = 0) {
    return this.r = t[e], this.g = t[e + 1], this.b = t[e + 2], this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this.r, t[e + 1] = this.g, t[e + 2] = this.b, t;
  }
  fromBufferAttribute(t, e) {
    return this.r = t.getX(e), this.g = t.getY(e), this.b = t.getZ(e), this;
  }
  toJSON() {
    return this.getHex();
  }
  *[Symbol.iterator]() {
    yield this.r, yield this.g, yield this.b;
  }
}
const A = /* @__PURE__ */ new kt();
kt.NAMES = re;
class Te extends U {
  constructor(t, e, s, i, n, h, a, r, o, l) {
    if (l = l !== void 0 ? l : ft, l !== ft && l !== Lt)
      throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");
    s === void 0 && l === ft && (s = pe), s === void 0 && l === Lt && (s = xe), super(null, i, n, h, a, r, l, s, o), this.isDepthTexture = !0, this.image = { width: t, height: e }, this.magFilter = a !== void 0 ? a : Bt, this.minFilter = r !== void 0 ? r : Bt, this.flipY = !1, this.generateMipmaps = !1, this.compareFunction = null;
  }
  copy(t) {
    return super.copy(t), this.compareFunction = t.compareFunction, this;
  }
  toJSON(t) {
    const e = super.toJSON(t);
    return this.compareFunction !== null && (e.compareFunction = this.compareFunction), e;
  }
}
const De = /* @__PURE__ */ new Te(1, 1);
De.compareFunction = fe;
typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register", { detail: {
  revision: te
} }));
typeof window < "u" && (window.__THREE__ ? console.warn("WARNING: Multiple instances of Three.js being imported.") : window.__THREE__ = te);
class Re {
  constructor() {
    b(this, "logging");
    b(this, "usedBefore", !1);
    b(this, "contentRef", "");
    b(this, "legacyMode", !1);
    b(this, "materialNames", /* @__PURE__ */ new Set());
    b(this, "modelName", "noname");
    b(this, "materialPerSmoothingGroup", !1);
    b(this, "useOAsMesh", !1);
    b(this, "useIndices", !1);
    b(this, "disregardNormals", !1);
    b(this, "vertices", []);
    b(this, "colors", []);
    b(this, "normals", []);
    b(this, "uvs", []);
    b(this, "rawMesh");
    b(this, "inputObjectCount", 1);
    b(this, "outputObjectCount", 1);
    b(this, "globalCounts");
    this.logging = this.buildDefaultLogging(), this.rawMesh = this.buildDefaultRawMesh(), this.globalCounts = this.buildDefaultGlobalsCount();
  }
  buildDefaultLogging() {
    return {
      enabled: !1,
      debug: !1
    };
  }
  buildDefaultRawMesh() {
    return {
      objectName: "",
      groupName: "",
      activeMtlName: "",
      mtllibName: "",
      // reset with new mesh
      faceType: -1,
      subGroups: /* @__PURE__ */ new Map(),
      subGroupInUse: void 0,
      smoothingGroup: {
        splitMaterials: !1,
        normalized: -1,
        real: -1
      },
      counts: {
        doubleIndicesCount: 0,
        faceCount: 0,
        mtlCount: 0,
        smoothingGroupCount: 0
      }
    };
  }
  buildDefaultGlobalsCount() {
    return {
      vertices: 0,
      faces: 0,
      doubleIndicesCount: 0,
      lineByte: 0,
      currentByte: 0,
      totalBytes: 0
    };
  }
  setBulkConfig(t) {
    this.materialPerSmoothingGroup = t.materialPerSmoothingGroup, this.useOAsMesh = t.useOAsMesh, this.useIndices = t.useIndices, this.disregardNormals = t.disregardNormals, this.modelName = t.modelName, this.materialNames = t.materialNames;
  }
  /**
   * Enable or disable logging in general (except warn and error), plus enable or disable debug logging.
   *
   * @param {boolean} enabled True or false.
   * @param {boolean} debug True or false.
   */
  setLogging(t, e) {
    this.logging.enabled = t === !0, this.logging.debug = e === !0;
  }
  setMaterialNames(t) {
    this.materialNames = t;
  }
  isLoggingEnabled() {
    return this.logging.enabled;
  }
  isDebugLoggingEnabled() {
    return this.logging.enabled && this.logging.debug;
  }
  /**
   *
   * @returns if parser was used before
   */
  isUsedBefore() {
    return this.usedBefore;
  }
  configure() {
    if (this.usedBefore = !0, this.pushSmoothingGroup("1"), this.logging.enabled) {
      const e = "OBJLoader2 Parser configuration:" + (this.materialNames.size > 0 ? `
	materialNames:
		- ` + Array.from(this.materialNames).join(`
		- `) : `
	materialNames: None`) + `
	materialPerSmoothingGroup: ` + this.materialPerSmoothingGroup + `
	useOAsMesh: ` + this.useOAsMesh + `
	useIndices: ` + this.useIndices + `
	disregardNormals: ` + this.disregardNormals;
      console.info(e);
    }
  }
  /**
   * Parse the provided arraybuffer
   *
   * @param {Uint8Array} arrayBuffer OBJ data as Uint8Array
   */
  execute(t) {
    this.logging.enabled && console.time("OBJLoader2Parser.execute"), this.configure();
    const e = new Uint8Array(t);
    this.contentRef = e;
    const s = e.byteLength;
    this.globalCounts.totalBytes = s;
    const i = new Array(128);
    let n = 0, h = 0, a = "", r = 0;
    for (let o; r < s; r++)
      switch (o = e[r], o) {
        case 32:
          a.length > 0 && (i[n++] = a), a = "";
          break;
        case 47:
          a.length > 0 && (i[n++] = a), h++, a = "";
          break;
        case 10:
          this.processLine(i, n, h, a, r), a = "", n = 0, h = 0;
          break;
        case 13:
          break;
        default:
          a += String.fromCharCode(o);
          break;
      }
    this.processLine(i, n, h, a, r), this.finalizeParsing(), this.logging.enabled && console.timeEnd("OBJLoader2Parser.execute");
  }
  /**
   * Parse the provided text
   *
   * @param {string} text OBJ data as string
   */
  executeLegacy(t) {
    this.logging.enabled && console.time("OBJLoader2Parser.executeLegacy"), this.configure(), this.legacyMode = !0, this.contentRef = t;
    const e = t.length;
    this.globalCounts.totalBytes = e;
    const s = new Array(128);
    let i = 0, n = 0, h = "", a = 0;
    for (let r; a < e; a++)
      switch (r = t[a], r) {
        case " ":
          h.length > 0 && (s[i++] = h), h = "";
          break;
        case "/":
          h.length > 0 && (s[i++] = h), n++, h = "";
          break;
        case `
`:
          this.processLine(s, i, n, h, a), h = "", i = 0, n = 0;
          break;
        case "\r":
          break;
        default:
          h += r;
      }
    this.processLine(s, i, n, h, a), this.finalizeParsing(), this.logging.enabled && console.timeEnd("OBJLoader2Parser.executeLegacy");
  }
  processLine(t, e, s, i, n) {
    if (this.globalCounts.lineByte = this.globalCounts.currentByte, this.globalCounts.currentByte = n, e < 1)
      return;
    i.length > 0 && (t[e++] = i);
    const h = (u, d, c, p) => {
      let x = "";
      if (p > c) {
        let g;
        if (d)
          for (g = c; g < p; g++)
            x += u[g];
        else
          for (g = c; g < p; g++)
            x += String.fromCharCode(u[g]);
        x = x.trim();
      }
      return x;
    };
    let a, r, o;
    const l = t[0];
    switch (l) {
      case "v":
        if (this.vertices.push(parseFloat(t[1])), this.vertices.push(parseFloat(t[2])), this.vertices.push(parseFloat(t[3])), e > 4) {
          const d = new kt();
          d.setRGB(
            parseFloat(t[4]),
            parseFloat(t[5]),
            parseFloat(t[6])
          ).convertSRGBToLinear(), this.colors.push(d.r), this.colors.push(d.g), this.colors.push(d.b);
        }
        break;
      case "vt":
        this.uvs.push(parseFloat(t[1])), this.uvs.push(parseFloat(t[2]));
        break;
      case "vn":
        this.normals.push(parseFloat(t[1])), this.normals.push(parseFloat(t[2])), this.normals.push(parseFloat(t[3]));
        break;
      case "f":
        if (a = e - 1, s === 0)
          for (this.checkFaceType(0), o = 2, r = a; o < r; o++)
            this.buildFace(t[1]), this.buildFace(t[o]), this.buildFace(t[o + 1]);
        else if (a === s * 2)
          for (this.checkFaceType(1), o = 3, r = a - 2; o < r; o += 2)
            this.buildFace(t[1], t[2]), this.buildFace(t[o], t[o + 1]), this.buildFace(t[o + 2], t[o + 3]);
        else if (a * 2 === s * 3)
          for (this.checkFaceType(2), o = 4, r = a - 3; o < r; o += 3)
            this.buildFace(t[1], t[2], t[3]), this.buildFace(t[o], t[o + 1], t[o + 2]), this.buildFace(t[o + 3], t[o + 4], t[o + 5]);
        else
          for (this.checkFaceType(3), o = 3, r = a - 2; o < r; o += 2)
            this.buildFace(t[1], void 0, t[2]), this.buildFace(t[o], void 0, t[o + 1]), this.buildFace(t[o + 2], void 0, t[o + 3]);
        break;
      case "l":
      case "p":
        if (a = e - 1, a === s * 2)
          for (this.checkFaceType(4), o = 1, r = a + 1; o < r; o += 2)
            this.buildFace(t[o], t[o + 1]);
        else
          for (this.checkFaceType(l === "l" ? 5 : 6), o = 1, r = a + 1; o < r; o++)
            this.buildFace(t[o]);
        break;
      case "s":
        this.pushSmoothingGroup(t[1]);
        break;
      case "g":
        this.processCompletedMesh(), this.rawMesh.groupName = h(this.contentRef, this.legacyMode, this.globalCounts.lineByte + 2, this.globalCounts.currentByte);
        break;
      case "o":
        this.useOAsMesh && this.processCompletedMesh(), this.rawMesh.objectName = h(this.contentRef, this.legacyMode, this.globalCounts.lineByte + 2, this.globalCounts.currentByte);
        break;
      case "mtllib":
        this.rawMesh.mtllibName = h(this.contentRef, this.legacyMode, this.globalCounts.lineByte + 7, this.globalCounts.currentByte);
        break;
      case "usemtl":
        const u = h(this.contentRef, this.legacyMode, this.globalCounts.lineByte + 7, this.globalCounts.currentByte);
        u !== "" && this.rawMesh.activeMtlName !== u && (this.rawMesh.activeMtlName = u, this.rawMesh.counts.mtlCount++, this.checkSubGroup());
        break;
    }
  }
  pushSmoothingGroup(t) {
    let e = parseInt(t);
    isNaN(e) && (e = t === "off" ? 0 : 1);
    const s = this.rawMesh.smoothingGroup.normalized;
    this.rawMesh.smoothingGroup.normalized = this.rawMesh.smoothingGroup.splitMaterials ? e : e === 0 ? 0 : 1, this.rawMesh.smoothingGroup.real = e, s !== e && (this.rawMesh.counts.smoothingGroupCount++, this.checkSubGroup());
  }
  /**
   * Expanded faceTypes include all four face types, both line types and the point type
   * faceType = 0: "f vertex ..."
   * faceType = 1: "f vertex/uv ..."
   * faceType = 2: "f vertex/uv/normal ..."
   * faceType = 3: "f vertex//normal ..."
   * faceType = 4: "l vertex/uv ..." or "l vertex ..."
   * faceType = 5: "l vertex ..."
   * faceType = 6: "p vertex ..."
   */
  checkFaceType(t) {
    this.rawMesh.faceType !== t && (this.processCompletedMesh(), this.rawMesh.faceType = t, this.checkSubGroup());
  }
  checkSubGroup() {
    const t = `${this.rawMesh.activeMtlName}|${this.rawMesh.smoothingGroup.normalized}`;
    this.rawMesh.subGroupInUse = this.rawMesh.subGroups.get(t), this.rawMesh.subGroupInUse || (this.rawMesh.subGroupInUse = {
      index: t,
      objectName: this.rawMesh.objectName,
      groupName: this.rawMesh.groupName,
      materialName: this.rawMesh.activeMtlName,
      smoothingGroup: this.rawMesh.smoothingGroup.normalized,
      vertices: [],
      indexMappingsCount: 0,
      indexMappings: /* @__PURE__ */ new Map(),
      indices: [],
      colors: [],
      uvs: [],
      normals: []
    }, this.rawMesh.subGroups.set(t, this.rawMesh.subGroupInUse));
  }
  buildFace(t, e, s) {
    const i = this.rawMesh.subGroupInUse, n = () => {
      const h = parseInt(t);
      let a = 3 * (h > 0 ? h - 1 : h + this.vertices.length / 3), r = this.colors.length > 0 ? a : null;
      const o = i.vertices;
      if (o.push(this.vertices[a++]), o.push(this.vertices[a++]), o.push(this.vertices[a]), r !== null) {
        const l = i.colors;
        l.push(this.colors[r++]), l.push(this.colors[r++]), l.push(this.colors[r]);
      }
      if (e) {
        const l = parseInt(e);
        let u = 2 * (l > 0 ? l - 1 : l + this.uvs.length / 2);
        const d = i.uvs;
        d.push(this.uvs[u++]), d.push(this.uvs[u]);
      }
      if (s && !this.disregardNormals) {
        const l = parseInt(s);
        let u = 3 * (l > 0 ? l - 1 : l + this.normals.length / 3);
        const d = i.normals;
        d.push(this.normals[u++]), d.push(this.normals[u++]), d.push(this.normals[u]);
      }
    };
    if (this.useIndices) {
      this.disregardNormals && (s = void 0);
      const h = t + (e ? "_" + e : "_n") + (s ? "_" + s : "_n");
      let a = i.indexMappings.get(h);
      a == null ? (a = this.rawMesh.subGroupInUse.vertices.length / 3, n(), i.indexMappings.set(h, a), i.indexMappingsCount++) : this.rawMesh.counts.doubleIndicesCount++, i.indices.push(a);
    } else
      n();
    this.rawMesh.counts.faceCount++;
  }
  createRawMeshReport(t) {
    return `Input Object number: ${t}
	Object name: ${this.rawMesh.objectName}
	Group name: ${this.rawMesh.groupName}
	Mtllib name: ${this.rawMesh.mtllibName}
	Vertex count: ${this.vertices.length / 3}
	Normal count: ${this.normals.length / 3}
	UV count: ${this.uvs.length / 2}
	SmoothingGroup count: ${this.rawMesh.counts.smoothingGroupCount}
	Material count: ${this.rawMesh.counts.mtlCount}
	Real MeshOutputGroup count: ${this.rawMesh.subGroups.size}`;
  }
  /**
   * Clear any empty subGroup and calculate absolute vertex, normal and uv counts
   */
  finalizeRawMesh() {
    const t = [];
    let e, s = 0, i = 0, n = 0, h = 0, a = 0, r = 0, o;
    for (const l of this.rawMesh.subGroups.entries())
      if (e = this.rawMesh.subGroups.get(l[0]), e && e.vertices.length > 0) {
        if (o = e.indices, o.length > 0 && i > 0)
          for (let u = 0; u < o.length; u++)
            o[u] = o[u] + i;
        t.push(e), s += e.vertices.length, i += e.indexMappingsCount, n += e.indices.length, h += e.colors.length, r += e.uvs.length, a += e.normals.length;
      }
    if (t.length > 0)
      return {
        name: this.rawMesh.groupName !== "" ? this.rawMesh.groupName : this.rawMesh.objectName,
        subGroups: t,
        absoluteVertexCount: s,
        absoluteIndexCount: n,
        absoluteColorCount: h,
        absoluteNormalCount: a,
        absoluteUvCount: r,
        faceCount: this.rawMesh.counts.faceCount,
        doubleIndicesCount: this.rawMesh.counts.doubleIndicesCount
      };
  }
  processCompletedMesh() {
    const t = this.finalizeRawMesh();
    if (t) {
      this.colors.length > 0 && this.colors.length !== this.vertices.length && this._onError("Vertex Colors were detected, but vertex count and color count do not match!"), this.logging.enabled && this.logging.debug && console.debug(this.createRawMeshReport(this.inputObjectCount)), this.inputObjectCount++;
      const e = this.createPreparedMesh(t);
      this._onAssetAvailable(e);
      const s = this.globalCounts.currentByte / this.globalCounts.totalBytes;
      return this._onProgress("Completed [o: " + this.rawMesh.objectName + " g:" + this.rawMesh.groupName + "] Total progress: " + (s * 100).toFixed(2) + "%"), this.resetRawMesh(), !0;
    }
    return !1;
  }
  resetRawMesh() {
    this.rawMesh.subGroups = /* @__PURE__ */ new Map(), this.rawMesh.subGroupInUse = void 0, this.rawMesh.smoothingGroup.normalized = -1, this.rawMesh.smoothingGroup.real = -1, this.pushSmoothingGroup("1"), this.rawMesh.counts.doubleIndicesCount = 0, this.rawMesh.counts.faceCount = 0, this.rawMesh.counts.mtlCount = 0, this.rawMesh.counts.smoothingGroupCount = 0;
  }
  /**
   * SubGroups are transformed to too intermediate format that is forwarded to the MeshReceiver.
   * It is ensured that SubGroups only contain objects with vertices (no need to check).
   *
   * @param result
   */
  createPreparedMesh(t) {
    const e = t.subGroups;
    if (this.globalCounts.vertices += t.absoluteVertexCount / 3, this.globalCounts.faces += t.faceCount, this.globalCounts.doubleIndicesCount += t.doubleIndicesCount, t.absoluteVertexCount <= 0)
      throw new Error(`Invalid vertex count: ${t.absoluteVertexCount}`);
    const s = new Float32Array(t.absoluteVertexCount), i = t.absoluteIndexCount > 0 ? new Uint32Array(t.absoluteIndexCount) : null, n = t.absoluteColorCount > 0 ? new Float32Array(t.absoluteColorCount) : null, h = t.absoluteNormalCount > 0 ? new Float32Array(t.absoluteNormalCount) : null, a = t.absoluteUvCount > 0 ? new Float32Array(t.absoluteUvCount) : null;
    let r, o = 0, l = 0, u = 0, d = 0, c = 0;
    const p = [];
    let x = 0, g = 0;
    const y = e.length > 1, C = [], w = n !== null;
    let _ = 0;
    const M = {
      materialCloneInstructions: [],
      materialName: "",
      multiMaterialNames: /* @__PURE__ */ new Map(),
      modelName: this.modelName,
      geometryType: this.rawMesh.faceType < 4 ? 0 : this.rawMesh.faceType === 6 ? 2 : 1
    };
    for (const F in e) {
      if (!Object.prototype.hasOwnProperty.call(e, F))
        continue;
      r = e[F];
      let f;
      const D = r.materialName, G = r.smoothingGroup === 0;
      this.rawMesh.faceType < 4 ? (f = D, w && (f += "_vertexColor"), G && (f += "_flat")) : f = this.rawMesh.faceType === 6 ? "defaultPointMaterial" : "defaultLineMaterial", M.materialName = f;
      const $ = this.materialNames.has(D), W = this.materialNames.has(f), q = !$ && !W, V = q ? !1 : !W;
      if (q && (f = w ? "defaultVertexColorMaterial" : "defaultMaterial", this.logging.enabled && console.info('object_group "' + r.objectName + "_" + r.groupName + '" was defined with unresolvable material "' + D + '"! Assigning "' + f + '".'), M.materialName = f), V) {
        const R = {
          materialNameOrg: D,
          materialProperties: {
            name: f,
            vertexColors: w ? 2 : 0,
            flatShading: G
          }
        };
        M.materialCloneInstructions.push(R);
      }
      if (y && (g = this.useIndices ? r.indices.length : r.vertices.length / 3, p.push({
        materialGroupOffset: x,
        materialGroupLength: g,
        materialIndex: _
      }), C[_] = f, M.multiMaterialNames.set(_, f), x += g, _++), s !== null && (s.set(r.vertices, o), o += r.vertices.length), i !== null && (i.set(r.indices, l), l += r.indices.length), n !== null && (n.set(r.colors, u), u += r.colors.length), h !== null && (h.set(r.normals, d), d += r.normals.length), a !== null && (a.set(r.uvs, c), c += r.uvs.length), this.logging.enabled && this.logging.debug) {
        let R = "";
        _ > 0 && (R = `
		materialIndex: ` + _);
        const Y = "	Output Object no.: " + this.outputObjectCount + `
		groupName: ` + r.groupName + `
		Index: ` + r.index + `
		faceType: ` + this.rawMesh.faceType + `
		materialName: ` + r.materialName + `
		smoothingGroup: ` + r.smoothingGroup + R + `
		objectName: ` + r.objectName + `
		#vertices: ` + r.vertices.length / 3 + `
		#indices: ` + r.indices.length + `
		#colors: ` + r.colors.length / 3 + `
		#uvs: ` + r.uvs.length / 2 + `
		#normals: ` + r.normals.length / 3;
        console.debug(Y);
      }
    }
    return this.outputObjectCount++, {
      meshName: t.name,
      vertexFA: s,
      normalFA: h,
      uvFA: a,
      colorFA: n,
      indexUA: i,
      createMultiMaterial: y,
      geometryGroups: p,
      multiMaterial: C,
      materialMetaInfo: M,
      progress: this.globalCounts.currentByte / this.globalCounts.totalBytes
    };
  }
  finalizeParsing() {
    if (this.logging.enabled && console.info("Global output object count: " + this.outputObjectCount), this.processCompletedMesh() && this.logging.enabled) {
      const t = `Overall counts: 
	Vertices: ` + this.globalCounts.vertices + `
	Faces: ` + this.globalCounts.faces + `
	Multiple definitions: ` + this.globalCounts.doubleIndicesCount;
      console.info(t);
    }
    this._onLoad();
  }
  /**
   * Announce parse progress feedback which is logged to the console.
   * @private
   *
   * @param {string} text Textual description of the event
   */
  _onProgress(t) {
    const e = t || "";
    this.logging.enabled && this.logging.debug && console.log(e);
  }
  /**
   * Announce error feedback which is logged as error message.
   * @private
   *
   * @param {String} errorMessage The event containing the error
   */
  _onError(t) {
    this.logging.enabled && this.logging.debug && console.error(t);
  }
  /**
   * Hook for alteration or transfer to main when parser is run in worker
   *
   * @param {Mesh} _mesh
   * @param {object} _materialMetaInfo
   */
  _onAssetAvailable(t, e) {
  }
  _onLoad() {
  }
}
class Oe {
  constructor() {
    b(this, "localData", {
      params: {},
      debugLogging: !1,
      materialNames: /* @__PURE__ */ new Set()
    });
  }
  initParser(t) {
    const e = new Re();
    return e._onAssetAvailable = (s) => {
      var a, r, o, l, u;
      const i = new Kt();
      i.message.params || (i.message.params = {}), i.message.params.preparedMesh = s, s.vertexFA !== null && ((a = i.message.buffers) == null || a.set("vertexFA", s.vertexFA)), s.normalFA !== null && ((r = i.message.buffers) == null || r.set("normalFA", s.normalFA)), s.uvFA !== null && ((o = i.message.buffers) == null || o.set("uvFA", s.uvFA)), s.colorFA !== null && ((l = i.message.buffers) == null || l.set("colorFA", s.colorFA)), s.indexUA !== null && ((u = i.message.buffers) == null || u.set("indexUA", s.indexUA));
      const n = T.createFromExisting(t, {
        overrideCmd: et.INTERMEDIATE_CONFIRM
      });
      n.addPayload(i), n.progress = s.progress;
      const h = T.pack(n.payloads, !1);
      self.postMessage(n, h);
    }, e._onLoad = () => {
      const s = T.createFromExisting(t, {
        overrideCmd: et.EXECUTE_COMPLETE
      });
      self.postMessage(s);
    }, e._onProgress = (s) => {
      e != null && e.isDebugLoggingEnabled() && console.debug("WorkerRunner: progress: " + s);
    }, e;
  }
  init(t) {
    const e = this.processMessage(t);
    this.localData.debugLogging && console.log(`OBJLoader2Worker#init: name: ${t.name} id: ${t.uuid} cmd: ${t.cmd} workerId: ${t.workerId}`);
    const s = T.createFromExisting(e, {
      overrideCmd: et.INIT_COMPLETE
    });
    self.postMessage(s);
  }
  execute(t) {
    this.processMessage(t);
    const e = this.initParser(t);
    Dt(e, this.localData.params, !1), this.localData.materialNames && (e == null || e.setMaterialNames(this.localData.materialNames)), e.isDebugLoggingEnabled() && console.log(`OBJLoader2Worker#execute: name: ${t.name} id: ${t.uuid} cmd: ${t.cmd} workerId: ${t.workerId}`), this.localData.buffer ? e.execute(this.localData.buffer) : self.postMessage(new Error("No ArrayBuffer was provided for parsing."));
  }
  processMessage(t) {
    var h, a;
    const e = T.unpack(t, !1), s = e.payloads[0];
    Dt(this.localData.params, s.message.params, !0);
    const i = (h = s.message.buffers) == null ? void 0 : h.get("modelData");
    i && (this.localData.buffer = i), s.message.params && s.message.params.materialNames && (this.localData.materialNames = s.message.params.materialNames);
    const n = ((a = this.localData.params) == null ? void 0 : a.logging) ?? {};
    return Object.hasOwn(n, "enabled") && Object.hasOwn(n, "debug") && (this.localData.debugLogging = n.enabled === !0 && n.debug === !0), e;
  }
}
const Be = new Oe();
self.onmessage = (m) => oe(Be, m);
