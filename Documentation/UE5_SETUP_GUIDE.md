# Setting up frame sending from UE5 to Python

This tutorial is about setting up video streaming from parking cameras from Unreal Engine to a website.
The system supports **2 cameras** (camera1 and camera2).

## Architecture

```
┌─────────────────────────────┐
│   Unreal Engine 5           │
│                             │
│  ┌───────────────────────┐  │
│  │ BP_Parking_Cam_1      │  │
│  │  → Scene Capture 2D   │  │
│  │  → RT_ParkingCamera1  │  │
│  │  → BP_FrameSender_1  ─┼──┼──→ HTTP POST (camera1)
│  └───────────────────────┘  │         ↓
│                             │
│  ┌───────────────────────┐  │
│  │ BP_Parking_Cam_2      │  │
│  │  → Scene Capture 2D   │  │
│  │  → RT_ParkingCamera2  │  │
│  │  → BP_FrameSender_2  ─┼──┼──→ HTTP POST (camera2)
│  └───────────────────────┘  │         ↓
└─────────────────────────────┘    ┌─────────────────┐
                                   │  Python Server   │
                                   │  (FastAPI)       │
                                   │    ↓             │
                                   │  YOLO Detection  │
                                   │    ↓             │
                                   │  WebSocket       │
                                   └────────┬─────────┘
                                            ↓
                                   ┌─────────────────┐
                                   │    Browser       │
                                   │  Camera 1: O|AI  │
                                   │  Camera 2: O|AI  │
                                   └─────────────────┘
```

## Step 1: Install the VaRest plugin

1. Open **Epic Games Launcher**
2. Go to the **Marketplace**
3. Find **"VaRest"** (Free HTTP request plugin)
4. Press **"Free"** -> **"Install to Engine"**
5. Relaunch Unreal Editor

## Step 2: Create Render Targets

Create **two** Render Targets — one per camera.

### Render Target 1 (Camera 1):
1. In **Content Browser** press **Add** -> **Materials & Textures** -> **Render Target**
2. Name: `RT_ParkingCamera1`
3. Set up parameters:
   - **Size X**: `1280`
   - **Size Y**: `720`
   - **Render Target Format**: `RTF RGBA8`
4. Save

### Render Target 2 (Camera 2):
1. Repeat the same steps
2. Name: `RT_ParkingCamera2`
3. Same parameters as above
4. Save

## Step 3: Set up parking cameras

### Camera 1 (BP_Parking_Cam_1):

1. Open Blueprint `BP_Parking_Cam_1`
2. Add component: **Add Component** -> **Scene Capture Component 2D**
3. Select this component and in **Details**:
   - **Texture Target**: choose `RT_ParkingCamera1`
   - **Capture Source**: `Final Color (LDR) in RGB`
   - **Primitive Render Mode**: `Render Scene Primitives`
4. Position the component so that it faces the parking lot

### Camera 2 (BP_Parking_Cam_2):

1. Create new Blueprint: **Content Browser** -> **Add** -> **Blueprint Class** -> **Actor**
2. Name: `BP_Parking_Cam_2`
3. Add components:
   - **Scene Capture Component 2D**
   - (optional) **Camera** for preview
4. Set up Scene Capture:
   - **Texture Target**: choose `RT_ParkingCamera2`
   - **Capture Source**: `Final Color (LDR) in RGB`
   - **Primitive Render Mode**: `Render Scene Primitives`
5. Position it to view a different area of the parking lot

## Step 4: Create Frame Sending Blueprints

You need **two** FrameSender actors — one per camera.

### 4.1 Create BP_FrameSender_1

1. **Content Browser** -> **Add** -> **Blueprint Class** -> **Actor**
2. Name: `BP_FrameSender_1`

#### Variables:

| Name | Type | Default Value |
|-----|-----|---------------|
| `ServerURL` | String | `http://localhost:8000/api/frame` |
| `RenderTarget` | Texture Render Target 2D (Object Reference) | - |
| `FrameRate` | Float | `15.0` |
| `bIsStreaming` | Boolean | `true` |
| `CameraID` | String | `camera1` |

#### Event Graph:

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
│ Export Render Target (RenderTarget, "C:/Temp/camera1.png")      │
│     ↓                                                           │
│ Load File to Array ("C:/Temp/camera1.png") → ByteArray          │
│     ↓                                                           │
│ Convert to Base64                                               │
│     ↓                                                           │
│ VaRest: Construct Json Request                                   │
│   - Add String Field: "camera_id" = CameraID                   │
│   - Add String Field: "frame_path" =                            │
│     "E:/Work/Computer_Vision/Projects/NeoSmart/                 │
│      SmartParking/frames/camera1.png"                           │
│     ↓                                                           │
│ VaRest: Apply URL (ServerURL)                                   │
│     ↓                                                           │
│ VaRest: Process URL (POST)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Create BP_FrameSender_2

1. Duplicate `BP_FrameSender_1` or create a new Blueprint
2. Name: `BP_FrameSender_2`
3. Change variables:
   - `CameraID`: `camera2`
4. In Event Graph change:
   - Export path: `"C:/Temp/camera2.png"`
   - frame_path in JSON: `".../frames/camera2.png"`

## Step 5: Place in level

1. Open your parking level
2. Drag `BP_FrameSender_1` into the scene
3. Select it and in **Details**:
   - **Render Target**: choose `RT_ParkingCamera1`
   - **Server URL**: `http://localhost:8000/api/frame`
   - **Frame Rate**: `15`
   - **CameraID**: `camera1`

4. Drag `BP_FrameSender_2` into the scene
5. Select it and in **Details**:
   - **Render Target**: choose `RT_ParkingCamera2`
   - **Server URL**: `http://localhost:8000/api/frame`
   - **Frame Rate**: `15`
   - **CameraID**: `camera2`

## Step 6: Launch

1. Start the Python server: `python main.py`
2. Open browser: `http://localhost:8000`
3. Press Play in Unreal Editor
4. Both camera streams should appear on the website!

## Troubleshooting

**Frames are not sent:**
- Make sure each Render Target is linked to its camera
- Check the paths to the files (must exist)
- Make sure the Python server is running
- Check that `camera_id` is set correctly in each BP_FrameSender

**Low FPS:**
- Reduce the resolution Render Target (640x480)
- Reduce Frame Rate to 10

**VaRest Errors:**
- Make sure the plugin is enabled in Project Settings -> Plugins

**Only one camera works:**
- Check that both BP_FrameSender actors are placed in the level
- Verify different CameraID values for each sender
- Make sure both Render Targets are assigned correctly
