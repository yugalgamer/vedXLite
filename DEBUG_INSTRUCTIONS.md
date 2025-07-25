# 🔧 Debug Instructions for AI Assistant

## ✅ How to Test the Enhanced Vision Feature

### 1. Start the Server
```bash
python app.py
```

### 2. Open Browser
Navigate to: `http://localhost:8000`

### 3. Test Enhanced Vision Tab

#### Step-by-Step:
1. **Click on "Enhanced Vision" tab** 
   - Should show status panel and upload area

2. **Check Gemma Status**
   - Should show: "Enhanced Gemma3n reasoning active (gemma3n:latest)"
   - Button should say: "Disable Enhanced Mode"

3. **Upload an Image** (Choose one method):
   
   **Method A: File Upload**
   - Click "Choose Image" button
   - Select any image file
   - **The "Analyze Image" button should appear** 

   **Method B: Camera Capture** 📷
   - Click "Live Camera" button  
   - Allow camera permissions
   - Click "Capture Image"
   - Click "Use This Image"
   - **The "Analyze Image" button should appear**

4. **Analyze the Image**
   - Click the **"Analyze Image"** button
   - Should show processing overlay
   - Should display analysis results

### 🐛 Common Issues & Solutions

#### Issue 1: "Analyze Image" button not appearing
**Cause**: Image preview not showing properly
**Solution**: 
- Check browser console for errors
- Make sure image file is valid (JPG/PNG)
- Try refreshing the page

#### Issue 2: "Enhanced reasoning unavailable"
**Cause**: Backend not updated or not running
**Solution**:
- Restart the Python server: `python app.py`
- Check console for "Gemma status" calls

#### Issue 3: Camera button not working
**Cause**: Function not globally available
**Solution**: 
- Check console for "startCameraCapture function assigned to window"
- Refresh the page hard (Ctrl+F5)

#### Issue 4: API 404/405 errors
**Cause**: Flask server endpoints not available
**Solution**:
- Make sure Flask server is running: `python app.py`
- Check server logs for endpoint requests

### 🔍 Debug Console Commands

Open browser DevTools (F12) and run:

```javascript
// Check if Enhanced Vision is initialized
console.log('Enhanced Vision Elements:', {
    analyzeBtn: !!document.getElementById('analyze-vision-btn'),
    uploadBtn: !!document.getElementById('vision-upload-btn'), 
    cameraBtn: !!document.getElementById('vision-camera-btn'),
    startCameraCapture: typeof window.startCameraCapture,
    analyzeVisionImage: typeof window.analyzeVisionImage
});

// Test camera function manually
if (window.startCameraCapture) {
    window.startCameraCapture();
} else {
    console.error('startCameraCapture not available');
}

// Check current image file
console.log('Current image:', window.currentImageFile || 'No image loaded');
```

### 📊 Server Endpoint Testing

Test endpoints manually:
```bash
# Test Gemma status
curl http://localhost:8000/api/gemma-status

# Test basic server status  
curl http://localhost:8000/api/status

# Test user endpoints
curl "http://localhost:8000/api/user-fetch?name=test"
```

### ✨ Expected Behavior

1. **Enhanced Vision Tab loads** ✅
2. **Gemma status shows "available" and "enabled"** ✅ 
3. **Camera button opens camera** ✅
4. **Upload button opens file picker** ✅
5. **After image upload/capture → "Analyze Image" button appears** ✅
6. **Clicking analyze → Shows processing → Shows results** ✅
7. **Quick action buttons work (Speak, Copy, Analyze Again)** ✅

If any step fails, check the browser console and server logs for error messages.
