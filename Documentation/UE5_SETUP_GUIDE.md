# Setting up frame sending from UE5 to Python

This tutorial is about setting up video streaming from parking cameras from Unreal Engine to a website.

## Architecture

```
┌─────────────────────────┐
│   Unreal Engine 5       │
│  ┌───────────────────┐  │
│  │ BP_Parking_Cam    │  │
│  │    ↓              │  │
│  │ Scene Capture 2D  │  │
│  │    ↓              │  │
│  │ Render Target     │  │
│  │    ↓              │  │
│  │ BP_FrameSender    │──┼──→ HTTP POST (JPEG)
│  └───────────────────┘  │         ↓
└─────────────────────────┘    ┌─────────────────┐
                               │  Python Server   │
                               │  (FastAPI)       │
                               │    ↓             │
                               │  YOLO Detection  │
                               │    ↓             │
                               │  WebSocket       │
                               └────────┬─────────┘
                                        ↓
                               ┌─────────────────┐
                               │    Browser      │
                               │  Original | AI  │
                               └─────────────────┘
```

## Step 1: Install the VaRest plugin

1. Open **Epic Games Launcher**
2. Go to the **Marketplace**
3. Find **"VaRest"** (Free HTTP request plugin)
4. Press **"Free"** -> **"Install to Engine"**
5. Relaunch Unreal Editor

## Step 2: Create Render Target

1. In **Content Browser** press **Add** -> **Materials & Textures** -> **Render Target**
2. Name: `RT_ParkingCamera`
3. Set up parameters:
   - **Size X**: `1280`
   - **Size Y**: `720`
   - **Render Target Format**: `RTF RGBA8`
4. Save

## Step 3: Set up a parking camera

### If you already have BP_Parking_Cam_1:

1. Open Blueprint `BP_Parking_Cam_1`
2. Add component: **Add Component** -> **Scene Capture Component 2D**
3. Select this component and in **Details**:
   - **Texture Target**: choose `RT_ParkingCamera`
   - **Capture Source**: `Final Color (LDR) in RGB`
   - **Primitive Render Mode**: `Render Scene Primitives`
4. Position the component so that it faces the parking lot (or bind it to the Camera component)

### If you create a new camera:

1. **Content Browser** -> **Add** -> **Blueprint Class** -> **Actor**
2. Name: `BP_ParkingCameraCapture`
3. Add components:
   - **Scene Capture Component 2D**
   - (optional) **Camera** для preview
4. Set up Scene Capture as above

## Step 4: Create a Frame Sending Blueprint

### 4.1 Create new Actor Blueprint

1. **Content Browser** -> **Add** -> **Blueprint Class** -> **Actor**
2. Name: `BP_FrameSender`

### 4.2 Add variables

In the **My Blueprint** -> **Variables** panel add:

| Name | Type | Default Value |
|-----|-----|---------------|
| `ServerURL` | String | `http://localhost:8000/api/frame` |
| `RenderTarget` | Texture Render Target 2D (Object Reference) | - |
| `FrameRate` | Float | `15.0` |
| `bIsStreaming` | Boolean | `true` |

### 4.3 Event Graph

```
┌─────────────────────────────────────────────────────────────────┐
│ Event BeginPlay                                                  │
│     ↓                                                           │
│ Set Timer by Function Name                                       │
│   - Function Name: "CaptureAndSendFrame"                        │
│   - Time: 1.0 / FrameRate  (≈ 0.066 for 15 FPS)                │
│   - Looping: true                                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Custom Event: CaptureAndSendFrame                                │
│     ↓                                                           │
│ Branch (bIsStreaming)                                           │
│   True →                                                        │
│     ↓                                                           │
│ Export Render Target (RenderTarget, "C:/Temp/frame.jpg")        │
│     ↓                                                           │
│ Load File to Array ("C:/Temp/frame.jpg") → ByteArray            │
│     ↓                                                           │
│ Convert to Base64                                               │
│     ↓                                                           │
│ VaRest: Construct Json Request                                   │
│   - Add String Field: "image" = Base64String                    │
│     ↓                                                           │
│ VaRest: Apply URL (ServerURL)                                   │
│     ↓                                                           │
│ VaRest: Process URL (POST)                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Шаг 5: Place in level

1. Unlock your own parking level
2. Drag `BP_FrameSender` into the scene
3. Select it and in **Details**:
   - **Render Target**: choose `RT_ParkingCamera`
   - **Server URL**: `http://localhost:8000/api/frame`
   - **Frame Rate**: `15`

## Step 6: Launch

1. Start the Python server: `python main.py`
2. Open browser: `http://localhost:8000`
3. Press Play in Unreal Editor
4. The video should appear on the website!

## Troubleshooting

**Frames are not sent:**
- Make sure the Render Target is linked to the camera.
- Check the path to the file (must exist)
- Make sure the Python server is running.

**Low FPS:**
- Reduce the resolution Render Target (640x480)
- Reduce Frame Rate to 10

**VaRest Errors:**
- Make sure the plugin is enabled in Project Settings -> Plugins
