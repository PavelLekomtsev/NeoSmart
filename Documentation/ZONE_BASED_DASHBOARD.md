# Zone-Based Smart Parking Dashboard

## Overview

The dashboard organizes the parking lot into two **zones** — Free Parking and Paid Parking — with a three-view navigation system. Each zone aggregates its own statistics independently, while the Overview page provides a combined summary.

This is a **frontend-only** feature. All cameras always stream simultaneously via WebSocket regardless of which view is active. Zone switching simply toggles CSS visibility — no backend changes were needed.

---

## Architecture

### Three-View System

![Zone Dashboard Architecture](extras/Zone_DashboardViews.png)

*The dashboard has three views: Overview (zone summary cards + global stats), Free Parking (camera1 + camera2 + zone stats), and Paid Parking (camera3 + zone stats + suspicious counter). Navigation tabs with live badges sit between the header and the content area.*

```
┌─────────────────────────────────────────────────────────┐
│  Header: Logo  |  Mode Selector  |  Connection Status   │
├─────────────────────────────────────────────────────────┤
│  [ Overview ]  [ Free Parking (5) ]  [ Paid Parking (3)]│
├─────────────────────────────────────────────────────────┤
│                                                         │
│                   Active View Content                   │
│             (only one view visible at a time)            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Views

| View | Cameras | Stats | Special |
|------|---------|-------|---------|
| **Overview** | None (summary only) | Global aggregate (all cameras) | Zone summary cards with mini-stats and progress bars |
| **Free Parking** | camera1, camera2 | Zone aggregate (camera1 + camera2) | Wrong parking count |
| **Paid Parking** | camera3 | Zone aggregate (camera3 only) | Suspicious cars counter, future: license plates, time tracking |

---

## Technologies

### WebSocket Streaming

The server (`main.py`) streams all camera frames over a single WebSocket connection (`/ws/stream`). Each message is a JSON object containing per-camera data:

```json
{
  "camera1": {
    "original": "<base64 JPEG>",
    "processed": "<base64 JPEG>",
    "stats": { "available": 3, "occupied": 5, "total_spaces": 8, ... }
  },
  "camera2": { ... },
  "camera3": { ... }
}
```

The browser receives all cameras every frame (~10 FPS). View switching does not affect the WebSocket — only CSS visibility changes.

### Canvas Rendering

Each camera has two HTML5 `<canvas>` elements:

- **Original** — raw frame from UE5 (no YOLO overlays)
- **Processed** — frame with YOLO bounding boxes, polygon overlays, and tracking visuals

Both canvases receive base64-encoded JPEG data. The JavaScript client decodes each to an `Image` object and draws it onto the canvas, automatically scaling to fit the container width.

**Canvas resize on view switch:** Hidden canvases report `clientWidth = 0`. When a view becomes visible, `resizeVisibleCanvases()` recalculates dimensions for all cameras in the newly shown view.

### CSS Animations

| Animation | Trigger | Effect |
|-----------|---------|--------|
| `viewFadeIn` | View becomes visible | Fade in + 8px upward slide (0.3s) |
| `statPulse` | Stat value changes | Brief scale 1→1.1→1 (0.3s) |

---

## Zone Configuration

Zones are defined in the `ParkingMonitor` JavaScript class:

```javascript
this.zones = {
    free: { cameras: ['camera1', 'camera2'], label: 'Free Parking' },
    paid: { cameras: ['camera3'], label: 'Paid Parking' }
};
```

Adding a new camera to a zone is a one-line change — add the camera ID to the `cameras` array, and the frontend automatically aggregates its stats into that zone.

### Zone Visual Identity

| Zone | Accent Color | CSS Variable | Usage |
|------|-------------|--------------|-------|
| Free Parking | Green `#22c55e` | `--zone-free-color` | Tab, card border, camera section top border, stats |
| Paid Parking | Amber `#f59e0b` | `--zone-paid-color` | Tab, card border, camera section top border, stats |
| Suspicious | Purple `#a855f7` | — | Suspicious stat card icon and count |

---

## Navigation

### Zone Tabs

Three pill-style buttons sit in a horizontal nav bar between the header and the content:

```html
<nav class="zone-nav">
    <button class="zone-tab active" data-view="overview">Overview</button>
    <button class="zone-tab" data-view="free">Free Parking <span class="zone-tab-badge">5</span></button>
    <button class="zone-tab" data-view="paid">Paid Parking <span class="zone-tab-badge">3</span></button>
</nav>
```

- The **active** tab gets a blue background at low opacity
- Badges show **live available spot counts** and update in real-time
- Clicking a tab calls `switchView(viewName)` which hides all views, shows the target, and re-triggers the fade animation
- On the Overview page, each zone summary card has a **"View Details →"** button that also calls `switchView()`

**Responsive behavior:** Below 600px, tab labels are hidden and only SVG icons are shown.

### View Switching Logic

```javascript
switchView(viewName) {
    // 1. Hide all views
    Object.values(this.views).forEach(v => v.classList.add('hidden'));

    // 2. Show target view + re-trigger animation
    const target = this.views[viewName];
    target.classList.remove('hidden');
    target.style.animation = 'none';
    target.offsetHeight;  // force reflow
    target.style.animation = '';

    // 3. Update active tab state
    document.querySelectorAll('.zone-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === viewName);
    });

    // 4. Resize canvases in the newly visible view
    this.resizeVisibleCanvases();
    this.currentView = viewName;
}
```

---

## Stats Aggregation

### Per-Zone Stats

Each zone computes its own aggregate stats by summing across its cameras:

```javascript
for (const camId of zone.cameras) {
    const stats = latestStats[camId];
    zoneStats.available += stats.available;
    zoneStats.occupied += stats.occupied;
    zoneStats.total += stats.total_spaces;
    zoneStats.wrong += stats.wrong_count;
    zoneStats.suspicious += stats.suspicious_count;
}
```

**Car count deduplication:** Camera1 and camera2 have overlapping fields of view. To avoid double-counting, only camera1's `cars_detected` is used for the free zone. Camera3 always counts for the paid zone.

```javascript
// Car counting — deduplicated
if (zoneName === 'free') {
    zoneStats.cars = latestStats['camera1']?.cars_detected || 0;
} else {
    zoneStats.cars = latestStats['camera3']?.cars_detected || 0;
}
```

### Global Aggregate (Overview)

The Overview page sums all zones' stats into a global aggregate shown in the stats panel and progress bar at the bottom.

### Stat Value Updates with Pulse

When a stat value changes, the `updateStatValue()` method updates the text and triggers a CSS pulse animation:

```javascript
updateStatValue(element, newValue) {
    const str = String(newValue);
    if (element.textContent !== str) {
        element.textContent = str;
        element.classList.remove('stat-updated');
        element.offsetHeight;  // force reflow
        element.classList.add('stat-updated');
    }
}
```

---

## Overview Page

The Overview page contains:

1. **Zone Summary Cards** — one per zone, arranged in a 2-column grid (1-column on mobile < 768px)
2. **Global Stats Panel** — aggregate available/occupied/total/cars/wrong counts
3. **Global Progress Bar** — overall occupancy percentage

### Zone Summary Card Structure

Each card has:
- **Colored left border** — green for free, amber for paid
- **Header** — zone name + camera count + "View Details →" button
- **Mini stats** — 3 compact stat indicators (Available, Occupied, Cars)
- **Mini progress bar** — zone occupancy percentage
- **Hover effect** — slight upward lift and shadow increase

---

## Files

| File | What Changed |
|------|-------------|
| `web_app/templates/index.html` | Zone nav, three view containers, overview cards, per-zone stats panels |
| `web_app/static/js/app.js` | `ParkingMonitor` class: zone config, `switchView()`, per-zone stat computation, overview cards, nav badges |
| `web_app/static/css/styles.css` | Zone nav, view system, overview grid, zone accents, animations, responsive rules |

### CSS Additions

| Component | Classes |
|-----------|---------|
| Zone nav bar | `.zone-nav`, `.zone-tab`, `.zone-tab.active`, `.zone-tab-badge` |
| View system | `.view`, `.view.hidden`, `@keyframes viewFadeIn` |
| Overview | `.overview-grid`, `.zone-summary-card`, `.zone-summary-header`, `.zone-mini-stat`, `.zone-mini-progress` |
| Zone accents | `#view-free .camera-section`, `#view-paid .camera-section` |
| Stat pulse | `@keyframes statPulse`, `.stat-updated` |

### HTML Element IDs (per zone)

| Zone | Available | Occupied | Total | Cars | Wrong/Suspicious | Progress |
|------|-----------|----------|-------|------|-------------------|----------|
| Free | `free-zone-available` | `free-zone-occupied` | `free-zone-total` | `free-zone-cars` | `free-zone-wrong` | `free-zone-progress-fill` |
| Paid | `paid-zone-available` | `paid-zone-occupied` | `paid-zone-total` | `paid-zone-cars` | `paid-zone-suspicious` | `paid-zone-progress-fill` |
| Overview (free card) | `overview-free-available` | `overview-free-occupied` | — | `overview-free-cars` | — | `overview-free-progress` |
| Overview (paid card) | `overview-paid-available` | `overview-paid-occupied` | — | `overview-paid-cars` | — | `overview-paid-progress` |

---

## Adding a New Zone

To add a new zone (e.g., "VIP Parking" with camera4):

1. **Backend:** Add `"camera4"` to `CAMERA_IDS` in `detector.py` and `main.py`
2. **Frontend JS:** Add zone entry:
   ```javascript
   this.zones.vip = { cameras: ['camera4'], label: 'VIP Parking' };
   ```
3. **Frontend HTML:**
   - Add a `<button class="zone-tab" data-view="vip">` in the nav
   - Add a `<div id="view-vip" class="view hidden">` with camera sections and stats panel
   - Add a zone summary card in the overview
4. **Frontend CSS:** Add accent color variable and zone-specific border rule
5. **Polygons:** Create `camera4_parkings.p` using `mark_parking_spaces.py`

---

## Verification

1. Start server: `cd SmartParking/web_app && python main.py`
2. Open `http://localhost:8000`
3. **Overview** — both zone summary cards display with live stats and mini progress bars
4. **Free Parking tab** — camera1 + camera2 visible, zone stats panel shows aggregate
5. **Paid Parking tab** — camera3 visible, suspicious counter active, zone stats panel shows aggregate
6. **Tab badges** — update in real-time with available spot counts
7. **Fullscreen** — works from any view (double-click or fullscreen button)
8. **Mode switch** — "Parking Spaces" / "Car Counter" works across all views
9. **Responsive** — at < 600px, tab labels hide and only icons show; at < 768px, overview cards stack vertically
