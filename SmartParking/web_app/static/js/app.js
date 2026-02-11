/**
 * Smart Parking Monitor - WebSocket Client
 */

class ParkingMonitor {
    constructor() {
        // Canvas elements
        this.originalCanvas = document.getElementById('original-canvas');
        this.processedCanvas = document.getElementById('processed-canvas');
        this.originalCtx = this.originalCanvas.getContext('2d');
        this.processedCtx = this.processedCanvas.getContext('2d');

        // Overlay elements
        this.originalOverlay = document.getElementById('original-overlay');
        this.processedOverlay = document.getElementById('processed-overlay');

        // Stats elements
        this.availableCount = document.getElementById('available-count');
        this.occupiedCount = document.getElementById('occupied-count');
        this.totalCount = document.getElementById('total-count');
        this.carsCount = document.getElementById('cars-count');
        this.occupancyPercent = document.getElementById('occupancy-percent');
        this.progressFill = document.getElementById('progress-fill');

        // Connection status
        this.connectionStatus = document.getElementById('connection-status');
        this.statusText = this.connectionStatus.querySelector('.status-text');

        // Source badge
        this.sourceBadge = document.getElementById('source-badge');

        // Frame path input
        this.framePathInput = document.getElementById('frame-path');
        this.setPathBtn = document.getElementById('set-path-btn');
        this.setPathBtn.addEventListener('click', () => this.setFramePath());

        // Mode selector
        this.modeSelect = document.getElementById('mode-select');
        this.modeSelect.addEventListener('change', () => this.changeMode());

        // WebSocket
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 2000;

        // Image cache for smoother rendering
        this.originalImage = new Image();
        this.processedImage = new Image();

        // Start connection
        this.connect();
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

        console.log('Connecting to:', wsUrl);
        this.updateConnectionStatus('connecting');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onclose = () => this.onClose();
            this.ws.onerror = (error) => this.onError(error);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.scheduleReconnect();
        }
    }

    onOpen() {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.updateConnectionStatus('connected');

        // Hide overlays
        this.originalOverlay.classList.add('hidden');
        this.processedOverlay.classList.add('hidden');
    }

    onMessage(event) {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'frame') {
                this.renderFrames(data.original, data.processed);
                this.updateStats(data.stats);
            }
        } catch (error) {
            console.error('Error processing message:', error);
        }
    }

    onClose() {
        console.log('WebSocket disconnected');
        this.updateConnectionStatus('disconnected');
        this.scheduleReconnect();
    }

    onError(error) {
        console.error('WebSocket error:', error);
        this.updateConnectionStatus('disconnected');
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting in ${this.reconnectDelay}ms... (attempt ${this.reconnectAttempts})`);

            setTimeout(() => this.connect(), this.reconnectDelay);
        } else {
            console.error('Max reconnect attempts reached');
            this.statusText.textContent = 'Connection failed';
        }
    }

    updateConnectionStatus(status) {
        this.connectionStatus.className = 'connection-status ' + status;

        switch (status) {
            case 'connected':
                this.statusText.textContent = 'Connected';
                break;
            case 'connecting':
                this.statusText.textContent = 'Connecting...';
                break;
            case 'disconnected':
                this.statusText.textContent = 'Disconnected';
                // Show overlays when disconnected
                this.originalOverlay.classList.remove('hidden');
                this.processedOverlay.classList.remove('hidden');
                break;
        }
    }

    renderFrames(originalB64, processedB64) {
        // Render original frame
        this.originalImage.onload = () => {
            this.resizeCanvas(this.originalCanvas, this.originalImage);
            this.originalCtx.drawImage(this.originalImage, 0, 0,
                this.originalCanvas.width, this.originalCanvas.height);
        };
        this.originalImage.src = 'data:image/jpeg;base64,' + originalB64;

        // Render processed frame
        this.processedImage.onload = () => {
            this.resizeCanvas(this.processedCanvas, this.processedImage);
            this.processedCtx.drawImage(this.processedImage, 0, 0,
                this.processedCanvas.width, this.processedCanvas.height);
        };
        this.processedImage.src = 'data:image/jpeg;base64,' + processedB64;
    }

    resizeCanvas(canvas, image) {
        // Get the container dimensions
        const container = canvas.parentElement;
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;

        // Calculate aspect ratio
        const imageRatio = image.width / image.height;
        const containerRatio = containerWidth / containerHeight;

        let width, height;

        if (imageRatio > containerRatio) {
            width = containerWidth;
            height = containerWidth / imageRatio;
        } else {
            height = containerHeight;
            width = containerHeight * imageRatio;
        }

        // Only resize if dimensions changed
        if (canvas.width !== Math.floor(width) || canvas.height !== Math.floor(height)) {
            canvas.width = Math.floor(width);
            canvas.height = Math.floor(height);
        }
    }

    updateStats(stats) {
        if (!stats) return;

        const total = stats.total_spaces || 0;
        const occupied = stats.occupied || 0;
        const available = stats.available || 0;
        const cars = stats.cars_detected || 0;

        // Update stat values
        this.availableCount.textContent = available;
        this.occupiedCount.textContent = occupied;
        this.totalCount.textContent = total;
        this.carsCount.textContent = cars;

        // Update progress bar
        const occupancyPercentage = total > 0 ? Math.round((occupied / total) * 100) : 0;
        this.occupancyPercent.textContent = occupancyPercentage + '%';
        this.progressFill.style.width = occupancyPercentage + '%';

        // Update mode selector if different
        if (stats.mode && this.modeSelect.value !== stats.mode) {
            this.modeSelect.value = stats.mode;
        }

        // Update source badge
        if (stats.frame_source) {
            this.updateSourceBadge(stats.ue5_connected ? stats.frame_source : 'none');
        }

        // Add visual feedback based on availability
        if (available === 0) {
            this.availableCount.style.color = '#ef4444'; // Red
        } else if (available <= 3) {
            this.availableCount.style.color = '#eab308'; // Yellow
        } else {
            this.availableCount.style.color = '#22c55e'; // Green
        }
    }

    changeMode() {
        const mode = this.modeSelect.value;

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'set_mode',
                mode: mode
            }));
            console.log('Mode changed to:', mode);
        }
    }

    setFramePath() {
        const path = this.framePathInput.value.trim();

        if (!path) {
            alert('Please enter a file path');
            return;
        }

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'set_watch_file',
                path: path
            }));
            console.log('Frame path set to:', path);

            // Visual feedback
            this.setPathBtn.textContent = 'Path Set!';
            setTimeout(() => {
                this.setPathBtn.textContent = 'Set Path';
            }, 2000);
        }
    }

    updateSourceBadge(source) {
        if (!this.sourceBadge) return;

        // Remove all source classes
        this.sourceBadge.classList.remove('http', 'file', 'screen');

        switch (source) {
            case 'http':
                this.sourceBadge.textContent = 'HTTP Stream';
                this.sourceBadge.classList.add('http');
                break;
            case 'file':
                this.sourceBadge.textContent = 'File Watcher';
                this.sourceBadge.classList.add('file');
                break;
            case 'screen':
                this.sourceBadge.textContent = 'Screen Capture';
                this.sourceBadge.classList.add('screen');
                break;
            default:
                this.sourceBadge.textContent = 'No Signal';
                break;
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.parkingMonitor = new ParkingMonitor();
});
