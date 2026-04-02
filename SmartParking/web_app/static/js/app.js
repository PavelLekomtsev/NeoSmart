/**
 * Smart Parking Monitor - WebSocket Client (Multi-Camera)
 */

class ParkingMonitor {
    constructor() {
        this.cameraIds = ['camera1', 'camera2'];

        // Per-camera elements
        this.cameras = {};
        for (const camId of this.cameraIds) {
            const originalCanvas = document.getElementById(`${camId}-original-canvas`);
            const processedCanvas = document.getElementById(`${camId}-processed-canvas`);

            this.cameras[camId] = {
                originalCanvas: originalCanvas,
                processedCanvas: processedCanvas,
                originalCtx: originalCanvas.getContext('2d'),
                processedCtx: processedCanvas.getContext('2d'),
                originalOverlay: document.getElementById(`${camId}-original-overlay`),
                processedOverlay: document.getElementById(`${camId}-processed-overlay`),
                statusBadge: document.getElementById(`${camId}-status`),
                availableEl: document.getElementById(`${camId}-available`),
                occupiedEl: document.getElementById(`${camId}-occupied`),
                carsEl: document.getElementById(`${camId}-cars`),
                wrongEl: document.getElementById(`${camId}-wrong`),
                wrongCard: document.getElementById(`${camId}-wrong-card`),
                originalImage: new Image(),
                processedImage: new Image(),
            };
        }

        // Aggregate stats elements
        this.availableCount = document.getElementById('available-count');
        this.occupiedCount = document.getElementById('occupied-count');
        this.totalCount = document.getElementById('total-count');
        this.carsCount = document.getElementById('cars-count');
        this.wrongCount = document.getElementById('wrong-count');
        this.wrongCard = document.getElementById('wrong-card');
        this.occupancyPercent = document.getElementById('occupancy-percent');
        this.progressFill = document.getElementById('progress-fill');

        // Connection status
        this.connectionStatus = document.getElementById('connection-status');
        this.statusText = this.connectionStatus.querySelector('.status-text');

        // Source badge
        this.sourceBadge = document.getElementById('source-badge');

        // Mode selector
        this.modeSelect = document.getElementById('mode-select');
        this.modeSelect.addEventListener('change', () => this.changeMode());

        // WebSocket
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 2000;

        // Fullscreen
        this.fullscreenModal = document.getElementById('fullscreen-modal');
        this.fullscreenCanvas = document.getElementById('fullscreen-canvas');
        this.fullscreenCtx = this.fullscreenCanvas.getContext('2d');
        this.fullscreenSourceCanvasId = null;

        document.getElementById('fullscreen-close').addEventListener('click', () => this.closeFullscreen());
        this.fullscreenModal.addEventListener('click', (e) => {
            if (e.target === this.fullscreenModal) this.closeFullscreen();
        });

        document.querySelectorAll('.fullscreen-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const canvasId = btn.dataset.canvas;
                this.openFullscreen(canvasId);
            });
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeFullscreen();
        });

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

        // Hide overlays for all cameras
        for (const camId of this.cameraIds) {
            const cam = this.cameras[camId];
            cam.originalOverlay.classList.add('hidden');
            cam.processedOverlay.classList.add('hidden');
        }
    }

    onMessage(event) {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'frame' && data.cameras) {
                let totalSpaces = 0;
                let totalOccupied = 0;
                let totalCars = 0;
                let totalWrong = 0;
                let anyConnected = false;
                let currentMode = null;

                for (const camId of this.cameraIds) {
                    const camData = data.cameras[camId];
                    if (camData) {
                        this.renderCameraFrames(camId, camData.original, camData.processed);
                        this.updateCameraStats(camId, camData.stats);

                        const stats = camData.stats || {};
                        totalSpaces += stats.total_spaces || 0;
                        totalOccupied += stats.occupied || 0;
                        totalWrong += stats.wrong_count || 0;
                        // Only count cars from camera1 (camera2 sees the same cars)
                        if (camId === 'camera1') {
                            totalCars += stats.cars_detected || 0;
                        }
                        if (stats.mode) currentMode = stats.mode;
                        if (stats.ue5_connected) anyConnected = true;
                    }
                }

                this.updateAggregateStats(totalSpaces, totalOccupied, totalCars, totalWrong, currentMode);
                this.updateSourceBadge(anyConnected ? 'http' : 'none');
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
                for (const camId of this.cameraIds) {
                    const cam = this.cameras[camId];
                    cam.originalOverlay.classList.remove('hidden');
                    cam.processedOverlay.classList.remove('hidden');
                }
                break;
        }
    }

    renderCameraFrames(camId, originalB64, processedB64) {
        const cam = this.cameras[camId];

        cam.originalImage.onload = () => {
            this.resizeCanvas(cam.originalCanvas, cam.originalImage);
            cam.originalCtx.drawImage(cam.originalImage, 0, 0,
                cam.originalCanvas.width, cam.originalCanvas.height);
        };
        cam.originalImage.src = 'data:image/jpeg;base64,' + originalB64;

        cam.processedImage.onload = () => {
            this.resizeCanvas(cam.processedCanvas, cam.processedImage);
            cam.processedCtx.drawImage(cam.processedImage, 0, 0,
                cam.processedCanvas.width, cam.processedCanvas.height);
            this.drawFullscreenFrame();
        };
        cam.processedImage.src = 'data:image/jpeg;base64,' + processedB64;
    }

    resizeCanvas(canvas, image) {
        const container = canvas.parentElement;
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;

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

        if (canvas.width !== Math.floor(width) || canvas.height !== Math.floor(height)) {
            canvas.width = Math.floor(width);
            canvas.height = Math.floor(height);
        }
    }

    updateCameraStats(camId, stats) {
        if (!stats) return;

        const cam = this.cameras[camId];
        const available = stats.available || 0;
        const occupied = stats.occupied || 0;
        const cars = stats.cars_detected || 0;
        const wrong = stats.wrong_count || 0;
        const connected = stats.ue5_connected || false;
        const isCarCounter = stats.mode === 'car_counter';

        cam.availableEl.textContent = available;
        cam.occupiedEl.textContent = isCarCounter ? '-' : occupied;
        cam.carsEl.textContent = cars;
        if (cam.wrongEl) cam.wrongEl.textContent = isCarCounter ? '-' : wrong;

        // Dim the occupied card when in car_counter mode
        const occupiedCard = cam.occupiedEl.closest('.camera-stat-card');
        if (occupiedCard) {
            occupiedCard.style.opacity = isCarCounter ? '0.4' : '1';
        }

        // Wrong card
        if (cam.wrongCard) {
            cam.wrongCard.style.opacity = isCarCounter ? '0.4' : '1';
            if (cam.wrongEl) {
                cam.wrongEl.style.color = (!isCarCounter && wrong > 0) ? '#ef4444' : '';
            }
        }

        // Update camera status badge
        if (connected) {
            cam.statusBadge.textContent = 'Online';
            cam.statusBadge.className = 'camera-status online';
        } else {
            cam.statusBadge.textContent = 'Offline';
            cam.statusBadge.className = 'camera-status offline';
        }

        // Color for available count
        if (available === 0) {
            cam.availableEl.style.color = '#ef4444';
        } else if (available <= 3) {
            cam.availableEl.style.color = '#eab308';
        } else {
            cam.availableEl.style.color = '#22c55e';
        }
    }

    updateAggregateStats(totalSpaces, totalOccupied, totalCars, totalWrong, mode) {
        const isCarCounter = mode === 'car_counter';
        const totalAvailable = totalSpaces - totalOccupied;

        this.availableCount.textContent = totalAvailable;
        this.occupiedCount.textContent = isCarCounter ? '-' : totalOccupied;
        this.totalCount.textContent = totalSpaces;
        this.carsCount.textContent = totalCars;
        if (this.wrongCount) this.wrongCount.textContent = isCarCounter ? '-' : totalWrong;

        const occupancyPercentage = totalSpaces > 0 ? Math.round((totalOccupied / totalSpaces) * 100) : 0;
        this.occupancyPercent.textContent = isCarCounter ? '-' : occupancyPercentage + '%';
        this.progressFill.style.width = isCarCounter ? '0%' : occupancyPercentage + '%';

        // Dim the aggregate occupied card in car_counter mode
        const occupiedCard = this.occupiedCount.closest('.stat-card');
        if (occupiedCard) {
            occupiedCard.style.opacity = isCarCounter ? '0.4' : '1';
        }

        // Wrong card
        if (this.wrongCard) {
            this.wrongCard.style.opacity = isCarCounter ? '0.4' : '1';
            if (this.wrongCount) {
                this.wrongCount.style.color = (!isCarCounter && totalWrong > 0) ? '#ef4444' : '';
            }
        }

        if (totalAvailable === 0) {
            this.availableCount.style.color = '#ef4444';
        } else if (totalAvailable <= 3) {
            this.availableCount.style.color = '#eab308';
        } else {
            this.availableCount.style.color = '#22c55e';
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

    openFullscreen(canvasId) {
        this.fullscreenSourceCanvasId = canvasId;
        this.fullscreenModal.classList.add('active');
        document.body.style.overflow = 'hidden';
        this.drawFullscreenFrame();
    }

    closeFullscreen() {
        this.fullscreenSourceCanvasId = null;
        this.fullscreenModal.classList.remove('active');
        document.body.style.overflow = '';
    }

    drawFullscreenFrame() {
        if (!this.fullscreenSourceCanvasId) return;

        // Find the Image object for this canvas to render at full resolution
        let sourceImage = null;
        for (const camId of this.cameraIds) {
            const cam = this.cameras[camId];
            if (this.fullscreenSourceCanvasId === `${camId}-original-canvas`) {
                sourceImage = cam.originalImage;
                break;
            } else if (this.fullscreenSourceCanvasId === `${camId}-processed-canvas`) {
                sourceImage = cam.processedImage;
                break;
            }
        }

        if (sourceImage && sourceImage.naturalWidth > 0) {
            this.fullscreenCanvas.width = sourceImage.naturalWidth;
            this.fullscreenCanvas.height = sourceImage.naturalHeight;
            this.fullscreenCtx.drawImage(sourceImage, 0, 0);
        }
    }

    updateSourceBadge(source) {
        if (!this.sourceBadge) return;

        this.sourceBadge.classList.remove('http', 'file', 'screen');

        switch (source) {
            case 'http':
                this.sourceBadge.textContent = 'HTTP Stream';
                this.sourceBadge.classList.add('http');
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
