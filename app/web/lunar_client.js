/**
 * LunarLander DAgger - Phase 1 Client
 * ES6 class handling Socket.IO binary protocol, keyboard input, and rendering.
 */

class LunarClient {

    // Binary protocol header layout (must match server)
    static HEADER_SIZE = 46;

    // Action mapping
    static ACTIONS = {
        ArrowLeft:  1,
        ArrowUp:    2,
        ArrowRight: 3,
        ArrowDown:  0,  // noop
    };

    static ACTION_ELEMENTS = {
        0: "actionNoop",
        1: "actionLeft",
        2: "actionMain",
        3: "actionRight",
    };

    static STATE_LABELS = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];

    constructor() {
        // Canvas
        this.canvas = document.getElementById("frameCanvas");
        this.ctx = this.canvas.getContext("2d");

        // UI elements
        this.connectionStatus = document.getElementById("connectionStatus");
        this.connectionText = this.connectionStatus.querySelector(".connection-text");
        this.overlay = document.getElementById("viewportOverlay");
        this.episodeStat = document.querySelector("#episodeStat .stat-value");
        this.stepStat = document.querySelector("#stepStat .stat-value");
        this.rewardStat = document.querySelector("#rewardStat .stat-value");
        this.historyList = document.getElementById("historyList");
        this.resetBtn = document.getElementById("resetBtn");

        // State
        this.connected = false;
        this.totalReward = 0;
        this.started = false;

        // Track which keys are currently held (prevents key repeat duplicates)
        this.heldKeys = new Set();

        this._initSocket();
        this._initKeyboard();
        this._initControls();
    }

    _initSocket() {
        const path = window.location.pathname.startsWith("/lunar")
            ? "/lunar/socket.io"
            : "/socket.io";

        this.socket = io({ path, transports: ["websocket"] });

        this.socket.on("connect", () => {
            this.connected = true;
            this.connectionStatus.classList.add("connected");
            this.connectionText.textContent = "Connected";
        });

        this.socket.on("disconnect", () => {
            this.connected = false;
            this.connectionStatus.classList.remove("connected");
            this.connectionText.textContent = "Disconnected";
            this.heldKeys.clear();
        });

        this.socket.on("frame", (data) => this._onFrame(data));
        this.socket.on("episode_end", (data) => this._onEpisodeEnd(data));
    }

    _initKeyboard() {
        document.addEventListener("keydown", (e) => {
            if (!this.connected) return;

            const action = LunarClient.ACTIONS[e.key];
            if (action === undefined) return;

            e.preventDefault();

            // Ignore key repeat events (key already held)
            if (this.heldKeys.has(e.key)) return;
            this.heldKeys.add(e.key);

            // Hide overlay on first input
            if (!this.started) {
                this.started = true;
                this.overlay.classList.add("hidden");
            }

            this._sendHeldActions();
        });

        document.addEventListener("keyup", (e) => {
            if (!this.connected) return;

            const action = LunarClient.ACTIONS[e.key];
            if (action === undefined) return;

            e.preventDefault();

            this.heldKeys.delete(e.key);

            this._sendHeldActions();
        });
    }

    _sendHeldActions() {
        // Collect all non-noop actions currently held
        const actions = [];
        for (const key of this.heldKeys) {
            const action = LunarClient.ACTIONS[key];
            if (action !== undefined && action !== 0) {
                actions.push(action);
            }
        }
        // Send array to server (empty array = noop)
        this.socket.emit("actions", actions);

        // Highlight: show first action or noop
        this._highlightAction(actions.length > 0 ? actions[0] : 0);
    }

    _initControls() {
        this.resetBtn.addEventListener("click", () => {
            if (!this.connected) return;
            this.totalReward = 0;
            this._updateRewardDisplay(0);
            this.heldKeys.clear();
            this._highlightAction(0);
            this.socket.emit("reset");
        });
    }

    _onFrame(data) {
        const buf = data instanceof ArrayBuffer ? data : data.buffer || data;
        const view = new DataView(buf);
        let offset = 0;

        // Read state vector (8 x float32 LE)
        const state = new Float32Array(8);
        for (let i = 0; i < 8; i++) {
            state[i] = view.getFloat32(offset, true);
            offset += 4;
        }

        // Action (uint8)
        const action = view.getUint8(offset);
        offset += 1;

        // Reward (float32 LE)
        const reward = view.getFloat32(offset, true);
        offset += 4;

        // Step (uint32 LE)
        const step = view.getUint32(offset, true);
        offset += 4;

        // Episode (uint32 LE)
        const episode = view.getUint32(offset, true);
        offset += 4;

        // Done (uint8)
        const done = view.getUint8(offset);
        offset += 1;

        // JPEG data: remainder of the buffer
        const jpegBytes = new Uint8Array(buf, offset);

        // Update UI
        this._updateState(state);
        this._updateStats(episode, step, reward);
        this._highlightAction(action);
        this._renderFrame(jpegBytes);
    }

    _onEpisodeEnd(data) {
        this._addHistoryItem(data.episode, data.steps, data.total_reward);
        this.totalReward = 0;
    }

    _updateState(state) {
        for (let i = 0; i < 8; i++) {
            const el = document.getElementById(LunarClient.STATE_LABELS[i]);
            if (el) {
                // Boolean display for leg contacts
                if (i >= 6) {
                    el.textContent = state[i] > 0.5 ? "1" : "0";
                } else {
                    el.textContent = state[i].toFixed(3);
                }
            }
        }
    }

    _updateStats(episode, step, reward) {
        this.totalReward += reward;
        this.episodeStat.textContent = episode;
        this.stepStat.textContent = step;
        this._updateRewardDisplay(this.totalReward);
    }

    _updateRewardDisplay(reward) {
        this.rewardStat.textContent = reward.toFixed(2);
        if (reward >= 200) {
            this.rewardStat.style.color = "var(--success)";
        } else if (reward < 0) {
            this.rewardStat.style.color = "var(--danger)";
        } else {
            this.rewardStat.style.color = "var(--text)";
        }
    }

    _highlightAction(action) {
        // Clear all
        document.querySelectorAll(".action-key").forEach((el) => {
            el.classList.remove("active");
        });

        // Highlight current
        const elId = LunarClient.ACTION_ELEMENTS[action];
        if (elId) {
            document.getElementById(elId).classList.add("active");
        }
    }

    _renderFrame(jpegBytes) {
        const blob = new Blob([jpegBytes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        const img = new window.Image();

        img.onload = () => {
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            URL.revokeObjectURL(url);
        };

        img.src = url;
    }

    _addHistoryItem(episode, steps, totalReward) {
        // Remove empty placeholder
        const empty = this.historyList.querySelector(".history-empty");
        if (empty) empty.remove();

        const item = document.createElement("div");
        item.className = "history-item";

        const rewardClass = totalReward >= 200 ? "positive" : totalReward < 0 ? "negative" : "";

        item.innerHTML = `
            <span class="history-ep">EP ${episode}</span>
            <span class="history-reward ${rewardClass}">${totalReward.toFixed(1)}</span>
            <span class="history-steps">${steps} steps</span>
        `;

        // Prepend (newest first)
        this.historyList.prepend(item);
    }
}

// Initialize on load
document.addEventListener("DOMContentLoaded", () => {
    window.lunarClient = new LunarClient();
});
