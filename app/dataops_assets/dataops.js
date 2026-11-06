/**
 * ComfyDL DataOps (dataops.js) - dynamic column-to-port binding for the
 * CdlDatasetPack / CdlDatasetUnpack node pair (comfydl/nodes/data_ops.py).
 *
 * What it does:
 *  - Pack: grows/shrinks the `col_N` input slots as the user connects and
 *    disconnects tensors.  The real column name rides in the slot's `label`
 *    (taken from the origin output's name at connect time); the serialized
 *    slot name stays `col_N`, which is what the backend's workflow
 *    reflection matches on (data_ops.reflect_input_names).
 *  - Unpack: rebuilds the output slots to mirror the table columns.  Two
 *    channels, both writing the same slot list:
 *      (a) static - when the dataset input connects to an upstream
 *          CdlDatasetPack (directly or through a cdlPack chain), mirror that
 *          Pack's input labels (List-Utils GODMT_Unpack pattern);
 *      (b) runtime - the backend returns `ui: {columns: [{name, type}]}`
 *          after each execution; onExecuted rebuilds the ports, and the
 *          rebuilt list persists inside the serialized workflow.
 *
 * Ported from ComfyUI-List-Utils (js/web.js L461-522) and
 * ComfyUI-basic_data_handling (web/js/dynamicnode.js), trimmed to what
 * pure forceInput slots need (no widget-type dynamic inputs, no groups).
 *
 * Containment: everything below that touches official internals goes through
 * `getApp()` (window.comfyAPI) and the LiteGraph node instance API
 * (addInput/addOutput/removeInput/removeOutput/onConnectionsChange/
 * onExecuted/onNodeCreated/app.configuringGraph).  Reduced-motion: this
 * module performs no animation of any kind.
 *
 * In: nothing (script tag).  Out: two registered node behaviour patches.
 */
(function () {
  "use strict";

  const TAG = "[ComfyDL dataops]";
  const COL_PREFIX = "col_";
  const PACK_CLASS = "CdlDatasetPack";
  const UNPACK_CLASS = "CdlDatasetUnpack";
  const DATASET_IO = "DATASET";
  const MAX_CHAIN = 20; // Pack->Unpack->... traversal cap (List-Utils uses 20)

  const getApp =
    () => (window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app) || null;

  // ------------------------------------------------------------- helpers ----

  function configuring(app) {
    // During graph load LiteGraph restores the slots from the serialized
    // workflow itself; reacting to the connection events fired along the way
    // would race the deserializer and drop links (List-Utils L237 guard).
    return !!(app && app.configuringGraph);
  }

  function nextColName(node) {
    // smallest unused col_N keeps the numbering stable after mid-list deletes
    let i = 0;
    while (node.inputs.some((s) => s.name === COL_PREFIX + i)) i += 1;
    return COL_PREFIX + i;
  }

  function renumber(node) {
    // Re-assign the serialized slot names col_0..N in slot order; labels
    // (the real column names) are preserved untouched.
    let i = 0;
    for (const slot of node.inputs) {
      slot.name = COL_PREFIX + i;
      i += 1;
    }
  }

  function isPlaceholder(node, slotIndex) {
    // The trailing slot is the "empty placeholder" the user connects next.
    return slotIndex === node.inputs.length - 1 &&
      node.inputs[slotIndex] &&
      !node.inputs[slotIndex].link;
  }

  function originOutputName(link_info, app) {
    const graph = app.graph;
    const origin = graph && graph.getNodeById && graph.getNodeById(link_info.origin_id);
    if (!origin) return null;
    const out = origin.outputs && origin.outputs[link_info.origin_slot];
    return out ? out.label || out.name : null;
  }

  function findPackOrigin(app, link_info) {
    // Follow the cdlPack chain upstream until a CdlDatasetPack is found
    // (Pack -> Unpack -> Pack -> ... chains are legal; List-Utils L96-124).
    let link = link_info;
    for (let hops = 0; link && hops < MAX_CHAIN; hops += 1) {
      const graph = app.graph;
      const node = graph && graph.getNodeById && graph.getNodeById(link.origin_id);
      if (!node) return null;
      if (node.comfyClass === PACK_CLASS) return node;
      const passthrough =
        node.inputs &&
        node.inputs.find((s) => s.type === DATASET_IO && s.link);
      if (!passthrough) return null;
      link = app.graph.links && app.graph.links[passthrough.link]
        ? app.graph.links[passthrough.link]
        : (app.graph._links && app.graph._links.get
          ? app.graph._links.get(passthrough.link)
          : null);
    }
    return null;
  }

  // ---------------------------------------------------------------- Pack ----

  function patchPack(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      if (!this.inputs || this.inputs.length === 0) {
        this.addInput(COL_PREFIX + "0", "TENSOR");
      }
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
      const app = getApp();
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined;
      if (!app || configuring(app)) return r;
      const isInput = slotType === 1 || slotType === "input";
      const connected = event === true || event === 1 || event === "connect";
      if (!isInput || !link_info) return r;

      if (connected) {
        // Stamp the real column name into the slot label (truth for the
        // backend reflection), then grow a fresh trailing placeholder.
        const name = originOutputName(link_info, app);
        if (name && this.inputs[slot]) this.inputs[slot].label = name;
        if (isPlaceholder(this, slot)) {
          this.addInput(nextColName(this), "TENSOR");
        }
      } else {
        // A mid-list disconnect: drop the slot and compact the numbering.
        if (this.inputs.length > 1 && slot < this.inputs.length) {
          this.removeInput(slot);
          renumber(this);
        }
      }
      return r;
    };
  }

  // -------------------------------------------------------------- Unpack ----

  function mirrorFromPack(node, packNode) {
    // Static channel: outputs = the Pack's connected inputs (labels first).
    const sources = (packNode.inputs || []).filter(
      (s) => s.link || s.label || s.name !== COL_PREFIX + (packNode.inputs.length - 1)
    );
    rebuildOutputs(node, sources.map((s) => ({
      name: s.label || s.name,
      type: s.type === DATASET_IO ? "*" : s.type || "TENSOR",
    })));
  }

  function rebuildOutputs(node, columns) {
    // Replace the whole output slot list with the given columns; existing
    // downstream links whose name/type survive the rebuild are re-attached
    // by slot index (LiteGraph keeps links alive across connectSide updates
    // only when the slot objects persist, so reuse slots when possible).
    const old = node.outputs || [];
    for (let i = old.length - 1; i >= columns.length; i -= 1) {
      node.removeOutput(i);
    }
    for (let i = 0; i < columns.length; i += 1) {
      const col = columns[i];
      if (i < old.length) {
        old[i].label = col.name;
        old[i].type = col.type;
        old[i].name = old[i].name || col.name;
      } else {
        node.addOutput(col.name, col.type).label = col.name;
      }
    }
  }

  function patchUnpack(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      if (!this.outputs || this.outputs.length === 0) {
        const o = this.addOutput(COL_PREFIX + "0", "*");
        o.label = COL_PREFIX + "0";
      }
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
      const app = getApp();
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined;
      if (!app || configuring(app)) return r;
      const isInput = slotType === 1 || slotType === "input";
      const connected = event === true || event === 1 || event === "connect";
      if (!isInput || !link_info || !connected) return r;
      const pack = findPackOrigin(app, link_info);
      if (pack) mirrorFromPack(this, pack);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
      // Runtime channel: the backend reports the real column schema after
      // each execution; the rebuilt ports serialize with the workflow, so
      // they survive save/load without re-running.
      if (message && Array.isArray(message.columns) && message.columns.length) {
        rebuildOutputs(this, message.columns);
      }
      return r;
    };
  }

  // ---------------------------------------------------------------- boot ----

  function start() {
    const app = getApp();
    if (!app || typeof app.registerExtension !== "function") return false;
    try {
      app.registerExtension({
        name: "ComfyDL.DataOps",
        beforeRegisterNodeDef(nodeType, nodeData) {
          if (nodeData && nodeData.name === PACK_CLASS) patchPack(nodeType);
          if (nodeData && nodeData.name === UNPACK_CLASS) patchUnpack(nodeType);
        },
      });
    } catch (e) {
      // The frontend's extension-service store may not be bootstrapped yet
      // (same race the crashsite/motion loaders gate on) — throwing here
      // would poison the host's bootstrap chain.  Bail and let the startup
      // interval retry once the stores exist. Expected on every cold boot,
      // so this is a debug-level note, not a warning worth alarming on.
      console.debug(TAG, "registerExtension too early, retrying:", e && e.message);
      return false;
    }
    console.log(TAG, "registered (Pack/Unpack dynamic column ports)");
    return true;
  }

  if (!start()) {
    let tries = 0;
    const timer = setInterval(() => {
      tries += 1;
      if (start() || tries > 150) {
        clearInterval(timer);
        if (tries > 150) console.warn(TAG, "comfyAPI never appeared; giving up");
      }
    }, 200);
  }
})();
