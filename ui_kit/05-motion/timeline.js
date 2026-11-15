/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Behaviour structure from framework TransformSequence (BeginAbsoluteSequence /
 * BeginDelayedSequence / Delay / Then) and Rulesets/UI/FrameStabilityContainer.cs
 * (seek / rewind must produce the same result). Rewritten as a native ES module.
 */

/**
 * A small absolute-time timeline. Events are keyed to an absolute millisecond
 * position, so seeking or rewinding replays exactly the same states — the same
 * property that makes upstream replay/status-pulse effects deterministic.
 */
export class Timeline {
  constructor(options = {}) {
    this.duration = Number(options.duration) || 0;
    this.clock = options.clock || (() => performance.now());
    this.onComplete = options.onComplete || null;
    this.rate = 1;
    this.events = [];
    this._lastTime = 0;
    this._playing = false;
    this._raf = 0;
    this._origin = 0;
    this._startedAt = 0;
    this._currentTime = 0;
    this._tick = this._tick.bind(this);
  }

  /** Schedule an action at an absolute time (BeginAbsoluteSequence equivalent). */
  at(time, fn) {
    const t = Math.max(0, Number(time) || 0);
    this.events.push({ time: t, fn, fired: false });
    this.events.sort((a, b) => a.time - b.time);
    this._lastTime = Math.max(this._lastTime, t);
    this.duration = Math.max(this.duration, t);
    return this;
  }

  /** Schedule an action relative to the last scheduled time (Delay/Then equivalent). */
  then(delay, fn) {
    return this.at(this._lastTime + (Number(delay) || 0), fn);
  }

  /** Bulk-load a sequence from 05-motion/motion.json. */
  sequence(steps, resolveAction) {
    for (const step of steps || []) {
      const fn = resolveAction(step);
      if (typeof fn === 'function') this.at(step.at, fn);
    }
    return this;
  }

  setRate(rate) {
    const next = Number(rate);
    if (!Number.isFinite(next) || next <= 0) return;
    if (this._playing) {
      this._origin = this._currentTime;
      this._startedAt = this.clock();
    }
    this.rate = next;
  }

  play(from = 0) {
    this.cancel();
    this._origin = Math.max(0, Number(from) || 0);
    this._startedAt = this.clock();
    this._currentTime = this._origin;
    this._resetFired(this._origin);
    this._playing = true;
    this._raf = requestAnimationFrame(this._tick);
    return this;
  }

  pause() {
    if (!this._playing) return;
    this._playing = false;
    cancelAnimationFrame(this._raf);
    this._origin = this._currentTime;
  }

  /** Seek without replaying events that are already in the past. */
  seek(ms) {
    const target = Math.max(0, Number(ms) || 0);
    this._currentTime = target;
    this._origin = target;
    this._startedAt = this.clock();
    this._resetFired(target);
    if (!this._playing) this._flush(target);
  }

  cancel() {
    this._playing = false;
    cancelAnimationFrame(this._raf);
    this._raf = 0;
    for (const event of this.events) event.fired = false;
  }

  get currentTime() { return this._currentTime; }
  get isPlaying() { return this._playing; }

  _resetFired(from) {
    for (const event of this.events) event.fired = event.time < from;
  }

  _flush(now) {
    for (const event of this.events) {
      if (!event.fired && event.time <= now) {
        event.fired = true;
        try { event.fn(event.time); } catch (error) { console.error('[timeline] action failed', error); }
      }
    }
  }

  _tick() {
    if (!this._playing) return;
    const elapsed = (this.clock() - this._startedAt) * this.rate;
    this._currentTime = this._origin + elapsed;
    this._flush(this._currentTime);

    if (this._currentTime >= this.duration) {
      this._playing = false;
      this._raf = 0;
      if (typeof this.onComplete === 'function') this.onComplete();
      return;
    }
    this._raf = requestAnimationFrame(this._tick);
  }
}

export function createTimeline(options) {
  return new Timeline(options);
}

export default Timeline;
