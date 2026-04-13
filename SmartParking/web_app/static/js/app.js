/**
 * Smart Parking Monitor - WebSocket Client (Multi-Camera, Zone-Based)
 */

class ParkingMonitor {
    constructor() {
        this.cameraIds = ['camera1', 'camera2', 'camera3', 'camera4'];

        // Zone definitions
        this.zones = {
            free: { cameras: ['camera1', 'camera2'], label: 'Free Parking' },
            paid: { cameras: ['camera3'], label: 'Paid Parking' },
            road: { cameras: ['camera4'], label: 'Road Traffic' }
        };

        // Current active view
        this.currentView = 'overview';

        // View elements
        this.views = {
            overview: document.getElementById('view-overview'),
            free: document.getElementById('view-free'),
            paid: document.getElementById('view-paid'),
            road: document.getElementById('view-road')
        };

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
                suspiciousEl: document.getElementById(`${camId}-suspicious`),
                suspiciousCard: document.getElementById(`${camId}-suspicious-card`),
                incomingEl: document.getElementById(`${camId}-incoming`),
                outgoingEl: document.getElementById(`${camId}-outgoing`),
                trafficTotalEl: document.getElementById(`${camId}-total`),
                originalImage: new Image(),
                processedImage: new Image(),
            };
        }

        // Global aggregate stats elements (overview)
        this.availableCount = document.getElementById('available-count');
        this.occupiedCount = document.getElementById('occupied-count');
        this.totalCount = document.getElementById('total-count');
        this.carsCount = document.getElementById('cars-count');
        this.wrongCount = document.getElementById('wrong-count');
        this.wrongCard = document.getElementById('wrong-card');
        this.occupancyPercent = document.getElementById('occupancy-percent');
        this.progressFill = document.getElementById('progress-fill');

        // Per-zone stats elements
        this.zoneElements = {
            free: {
                available: document.getElementById('free-zone-available'),
                occupied: document.getElementById('free-zone-occupied'),
                total: document.getElementById('free-zone-total'),
                cars: document.getElementById('free-zone-cars'),
                wrong: document.getElementById('free-zone-wrong'),
                wrongCard: document.getElementById('free-zone-wrong-card'),
                progressFill: document.getElementById('free-zone-progress-fill'),
                occupancyPercent: document.getElementById('free-zone-occupancy-percent'),
            },
            paid: {
                available: document.getElementById('paid-zone-available'),
                occupied: document.getElementById('paid-zone-occupied'),
                total: document.getElementById('paid-zone-total'),
                cars: document.getElementById('paid-zone-cars'),
                wrong: document.getElementById('paid-zone-wrong'),
                wrongCard: document.getElementById('paid-zone-wrong-card'),
                suspicious: document.getElementById('paid-zone-suspicious'),
                suspiciousCard: document.getElementById('paid-zone-suspicious-card'),
                progressFill: document.getElementById('paid-zone-progress-fill'),
                occupancyPercent: document.getElementById('paid-zone-occupancy-percent'),
            },
            road: {
                incoming: document.getElementById('road-zone-incoming'),
                outgoing: document.getElementById('road-zone-outgoing'),
                total: document.getElementById('road-zone-total'),
            }
        };

        // Overview card elements
        this.overviewElements = {
            free: {
                available: document.getElementById('overview-free-available'),
                occupied: document.getElementById('overview-free-occupied'),
                cars: document.getElementById('overview-free-cars'),
                progress: document.getElementById('overview-free-progress'),
            },
            paid: {
                available: document.getElementById('overview-paid-available'),
                occupied: document.getElementById('overview-paid-occupied'),
                cars: document.getElementById('overview-paid-cars'),
                suspicious: document.getElementById('overview-paid-suspicious'),
                progress: document.getElementById('overview-paid-progress'),
            },
            road: {
                incoming: document.getElementById('overview-road-incoming'),
                outgoing: document.getElementById('overview-road-outgoing'),
                total: document.getElementById('overview-road-total'),
            }
        };

        // Nav badge elements
        this.navBadges = {
            free: document.getElementById('free-zone-count'),
            paid: document.getElementById('paid-zone-count'),
            road: document.getElementById('road-zone-count'),
        };

        // Connection status
        this.connectionStatus = document.getElementById('connection-status');
        this.statusText = this.connectionStatus.querySelector('.status-text');

        // Mode selector
        this.modeSelect = document.getElementById('mode-select');
        this.modeSelect.addEventListener('change', () => this.changeMode());

        // Zone tab navigation
        document.querySelectorAll('.zone-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchView(tab.dataset.view));
        });

        // Zone detail buttons (on overview cards)
        document.querySelectorAll('.zone-detail-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchView(btn.dataset.view));
        });

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

        // Restore last active view from localStorage
        const savedView = localStorage.getItem('parkingMonitor_activeView');
        if (savedView && this.views[savedView]) {
            this.switchView(savedView);
        }

        // Start connection
        this.connect();
    }

    // --- View Switching ---

    switchView(viewName) {
        if (!this.views[viewName] || viewName === this.currentView) return;

        // Hide all views
        Object.values(this.views).forEach(v => v.classList.add('hidden'));

        // Show target view with animation
        const target = this.views[viewName];
        target.classList.remove('hidden');
        target.style.animation = 'none';
        target.offsetHeight; // force reflow
        target.style.animation = '';

        // Update tab active state
        document.querySelectorAll('.zone-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.view === viewName);
        });

        this.currentView = viewName;
        localStorage.setItem('parkingMonitor_activeView', viewName);

        // Resize canvases in newly visible view (they had 0 size when hidden)
        this.resizeVisibleCanvases(viewName);
    }

    resizeVisibleCanvases(viewName) {
        const zoneDef = this.zones[viewName];
        if (!zoneDef) return;

        for (const camId of zoneDef.cameras) {
            const cam = this.cameras[camId];
            if (cam.originalImage.naturalWidth > 0) {
                this.resizeCanvas(cam.originalCanvas, cam.originalImage);
                cam.originalCtx.drawImage(cam.originalImage, 0, 0,
                    cam.originalCanvas.width, cam.originalCanvas.height);
            }
            if (cam.processedImage.naturalWidth > 0) {
                this.resizeCanvas(cam.processedCanvas, cam.processedImage);
                cam.processedCtx.drawImage(cam.processedImage, 0, 0,
                    cam.processedCanvas.width, cam.processedCanvas.height);
            }
        }
    }

    // --- WebSocket ---

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

                // Per-zone accumulators
                const zoneStats = {};
                for (const zoneName of Object.keys(this.zones)) {
                    zoneStats[zoneName] = { available: 0, occupied: 0, cars: 0, wrong: 0, suspicious: 0, total: 0, incoming: 0, outgoing: 0 };
                }

                for (const camId of this.cameraIds) {
                    const camData = data.cameras[camId];
                    if (camData) {
                        this.renderCameraFrames(camId, camData.original, camData.processed);
                        this.updateCameraStats(camId, camData.stats);

                        const stats = camData.stats || {};
                        const spaces = stats.total_spaces || 0;
                        const occupied = stats.occupied || 0;
                        const wrong = stats.wrong_count || 0;
                        const cars = stats.cars_detected || 0;

                        // Road cameras don't contribute to parking totals
                        if (camId !== 'camera4') {
                            totalSpaces += spaces;
                            totalOccupied += occupied;
                            totalWrong += wrong;
                            // Only count cars from camera1 (camera2 sees same area)
                            if (camId === 'camera1') {
                                totalCars += cars;
                            }
                            // camera3 (paid zone) always counts its own cars
                            if (camId === 'camera3') {
                                totalCars += cars;
                            }
                        }

                        if (stats.mode) currentMode = stats.mode;
                        if (stats.ue5_connected) anyConnected = true;

                        // Accumulate into zone stats
                        const suspicious = stats.suspicious_count || 0;
                        const incoming = stats.incoming_count || 0;
                        const outgoing = stats.outgoing_count || 0;

                        for (const [zoneName, zoneDef] of Object.entries(this.zones)) {
                            if (zoneDef.cameras.includes(camId)) {
                                zoneStats[zoneName].total += spaces;
                                zoneStats[zoneName].occupied += occupied;
                                zoneStats[zoneName].wrong += wrong;
                                zoneStats[zoneName].suspicious += suspicious;
                                zoneStats[zoneName].incoming += incoming;
                                zoneStats[zoneName].outgoing += outgoing;
                                // Dedup: only camera1 for free zone cars
                                if (zoneName === 'free' && camId === 'camera1') {
                                    zoneStats[zoneName].cars += cars;
                                }
                                if (zoneName === 'paid') {
                                    zoneStats[zoneName].cars += cars;
                                }
                                if (zoneName === 'road') {
                                    zoneStats[zoneName].cars += cars;
                                }
                            }
                        }
                    }
                }

                // Compute available for each zone
                for (const zoneName of Object.keys(zoneStats)) {
                    zoneStats[zoneName].available = zoneStats[zoneName].total - zoneStats[zoneName].occupied;
                }

                // Update all UI sections
                this.updateAggregateStats(totalSpaces, totalOccupied, totalCars, totalWrong, currentMode);
                this.updateZoneStats('free', zoneStats.free, currentMode);
                this.updateZoneStats('paid', zoneStats.paid, currentMode);
                this.updateZoneStats('road', zoneStats.road, currentMode);
                this.updateOverviewCards(zoneStats, currentMode);
                this.updateNavBadges(zoneStats);
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

    // --- Rendering ---

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

        if (containerWidth === 0 || containerHeight === 0) return;

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

    // --- Stats Updates ---

    updateStatValue(element, newValue) {
        if (!element) return;
        const newStr = String(newValue);
        if (element.textContent !== newStr) {
            element.textContent = newStr;
            element.classList.remove('stat-updated');
            void element.offsetHeight;
            element.classList.add('stat-updated');
        }
    }

    updateCameraStats(camId, stats) {
        if (!stats) return;

        const cam = this.cameras[camId];
        const available = stats.available || 0;
        const occupied = stats.occupied || 0;
        const cars = stats.cars_detected || 0;
        const wrong = stats.wrong_count || 0;
        const suspicious = stats.suspicious_count || 0;
        const incoming = stats.incoming_count || 0;
        const outgoing = stats.outgoing_count || 0;
        const connected = stats.ue5_connected || false;
        const isCarCounter = stats.mode === 'car_counter';

        this.updateStatValue(cam.availableEl, available);
        this.updateStatValue(cam.occupiedEl, isCarCounter ? '-' : occupied);
        this.updateStatValue(cam.carsEl, cars);
        if (cam.wrongEl) this.updateStatValue(cam.wrongEl, isCarCounter ? '-' : wrong);
        if (cam.suspiciousEl) this.updateStatValue(cam.suspiciousEl, suspicious);
        this.updateStatValue(cam.incomingEl, incoming);
        this.updateStatValue(cam.outgoingEl, outgoing);
        this.updateStatValue(cam.trafficTotalEl, incoming + outgoing);

        const occupiedCard = cam.occupiedEl ? cam.occupiedEl.closest('.camera-stat-card') : null;
        if (occupiedCard) {
            occupiedCard.style.opacity = isCarCounter ? '0.4' : '1';
        }

        if (cam.wrongCard) {
            cam.wrongCard.style.opacity = isCarCounter ? '0.4' : '1';
            if (cam.wrongEl) {
                cam.wrongEl.style.color = (!isCarCounter && wrong > 0) ? '#ef4444' : '';
            }
        }

        if (cam.suspiciousCard) {
            if (cam.suspiciousEl) {
                cam.suspiciousEl.style.color = suspicious > 0 ? '#ef4444' : '';
            }
        }

        if (connected) {
            cam.statusBadge.textContent = 'Online';
            cam.statusBadge.className = 'camera-status online';
        } else {
            cam.statusBadge.textContent = 'Offline';
            cam.statusBadge.className = 'camera-status offline';
        }

        if (cam.availableEl) {
            if (available === 0) {
                cam.availableEl.style.color = '#ef4444';
            } else if (available <= 3) {
                cam.availableEl.style.color = '#eab308';
            } else {
                cam.availableEl.style.color = '#22c55e';
            }
        }
    }

    updateAggregateStats(totalSpaces, totalOccupied, totalCars, totalWrong, mode) {
        const isCarCounter = mode === 'car_counter';
        const totalAvailable = totalSpaces - totalOccupied;

        this.updateStatValue(this.availableCount, totalAvailable);
        this.updateStatValue(this.occupiedCount, isCarCounter ? '-' : totalOccupied);
        this.updateStatValue(this.totalCount, totalSpaces);
        this.updateStatValue(this.carsCount, totalCars);
        if (this.wrongCount) this.updateStatValue(this.wrongCount, isCarCounter ? '-' : totalWrong);

        const occupancyPercentage = totalSpaces > 0 ? Math.round((totalOccupied / totalSpaces) * 100) : 0;
        this.occupancyPercent.textContent = isCarCounter ? '-' : occupancyPercentage + '%';
        this.progressFill.style.width = isCarCounter ? '0%' : occupancyPercentage + '%';

        const occupiedCard = this.occupiedCount.closest('.stat-card');
        if (occupiedCard) {
            occupiedCard.style.opacity = isCarCounter ? '0.4' : '1';
        }

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

    updateZoneStats(zoneName, stats, mode) {
        const els = this.zoneElements[zoneName];
        if (!els) return;

        const isCarCounter = mode === 'car_counter';
        const { available, occupied, total, cars, wrong, suspicious, incoming, outgoing } = stats;

        this.updateStatValue(els.available, available);
        this.updateStatValue(els.occupied, isCarCounter ? '-' : occupied);
        this.updateStatValue(els.total, total);
        this.updateStatValue(els.cars, cars);
        if (els.wrong) this.updateStatValue(els.wrong, isCarCounter ? '-' : wrong);
        if (els.suspicious) this.updateStatValue(els.suspicious, suspicious);
        if (els.incoming) this.updateStatValue(els.incoming, incoming);
        if (els.outgoing) this.updateStatValue(els.outgoing, outgoing);

        const occupancyPct = total > 0 ? Math.round((occupied / total) * 100) : 0;
        if (els.occupancyPercent) {
            els.occupancyPercent.textContent = isCarCounter ? '-' : occupancyPct + '%';
        }
        if (els.progressFill) {
            els.progressFill.style.width = isCarCounter ? '0%' : occupancyPct + '%';
        }

        // Dim occupied and wrong cards in car_counter mode
        if (els.occupied) {
            const card = els.occupied.closest('.stat-card');
            if (card) card.style.opacity = isCarCounter ? '0.4' : '1';
        }
        if (els.wrongCard) {
            els.wrongCard.style.opacity = isCarCounter ? '0.4' : '1';
            if (els.wrong) {
                els.wrong.style.color = (!isCarCounter && wrong > 0) ? '#ef4444' : '';
            }
        }
        if (els.suspiciousCard) {
            if (els.suspicious) {
                els.suspicious.style.color = suspicious > 0 ? '#ef4444' : '';
            }
        }

        // Color for available
        if (els.available) {
            if (available === 0) {
                els.available.style.color = '#ef4444';
            } else if (available <= 3) {
                els.available.style.color = '#eab308';
            } else {
                els.available.style.color = '#22c55e';
            }
        }
    }

    updateOverviewCards(zoneStats, mode) {
        const isCarCounter = mode === 'car_counter';

        for (const [zoneName, stats] of Object.entries(zoneStats)) {
            const els = this.overviewElements[zoneName];
            if (!els) continue;

            this.updateStatValue(els.available, stats.available);
            this.updateStatValue(els.occupied, isCarCounter ? '-' : stats.occupied);
            this.updateStatValue(els.cars, stats.cars);
            if (els.suspicious) {
                this.updateStatValue(els.suspicious, stats.suspicious);
                els.suspicious.style.color = stats.suspicious > 0 ? '#ef4444' : '';
            }
            if (els.incoming) this.updateStatValue(els.incoming, stats.incoming);
            if (els.outgoing) this.updateStatValue(els.outgoing, stats.outgoing);
            if (els.total) this.updateStatValue(els.total, stats.incoming + stats.outgoing);

            // Mini progress bar
            const pct = stats.total > 0 ? Math.round((stats.occupied / stats.total) * 100) : 0;
            if (els.progress) {
                els.progress.style.width = isCarCounter ? '0%' : pct + '%';
            }

            // Color available value
            if (els.available) {
                if (stats.available === 0) {
                    els.available.style.color = '#ef4444';
                } else if (stats.available <= 3) {
                    els.available.style.color = '#eab308';
                } else {
                    // Use zone-specific color
                    els.available.style.color = zoneName === 'paid' ? '#f59e0b' : '#22c55e';
                }
            }
        }
    }

    updateNavBadges(zoneStats) {
        for (const [zoneName, stats] of Object.entries(zoneStats)) {
            const badge = this.navBadges[zoneName];
            if (badge) {
                // Road zone shows total traffic count, parking zones show available
                this.updateStatValue(badge, zoneName === 'road' ? (stats.incoming + stats.outgoing) : stats.available);
            }
        }
    }

    // --- Mode ---

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

    // --- Fullscreen ---

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
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.parkingMonitor = new ParkingMonitor();
});
