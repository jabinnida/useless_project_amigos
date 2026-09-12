<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />

# SAMOSA TRIANGLE ACCURACY FINDER 🎯

> *"Is it a samosa or a suspicious triangle? Because even samosas deserve a perfect score. No bias, just triangles."*

---

## Basic Details
### Team Name: Amigos

### Team Members
- **Team Lead:** VS Abhinav - NSSCE
- **Member 2:** Nida Jabin P - NSSCE


---

### Project Description
An over-engineered computer vision tool that rigorously evaluates the geometric perfection of samosas. It isolates the samosa from plates, tables, and hands, extracts its true 3D outer vertices, overlays a neon analysis triangle directly onto your original photo, and calculates its equilateral accuracy score using custom geometric formulas.

---

### The Problem (that doesn't exist)
For generations, humanity has consumed billions of samosas with complete disregard for Euclidean geometry. People routinely accept squashed, lopsided, or trapezoidal samosas without any objective metric of geometric justice. How do you know if your evening snack is truly an equilateral masterpiece or a culinary fraud?

---

### The Solution (that nobody asked for)
**Samosa Triangle Accuracy Finder** brings mathematical rigor to street food:
1. **Autonomous Background Removal:** Isolates the fried pastry from plates, tables, and fingers behind the scenes.
2. **Sub-Pixel Corner Detection:** Identifies the Apex (A), Bottom-Left (B), and Bottom-Right (C) physical corners.
3. **Equilateral Geometry Scoring:** Calculates side lengths, interior angles, and accuracy percentage based on:
   $$\text{Accuracy} = \left(1 - \frac{|a - b| + |b - c| + |c - a|}{3 \times s}\right) \times 100$$
4. **Imposter Rejection Engine:** Rejects non-samosa images (circles, rectangular books, keyboards, faces, coffee cups) using Ramer-Douglas-Peucker shape simplification, corner sharpness analysis, and contour IoU matching.

---

## Technical Details

### Technologies / Components Used
**For Software:**
- **Languages:** HTML5, CSS3, JavaScript (ES6+)
- **APIs & Frameworks:** HTML5 Canvas API, WebRTC `mediaDevices` Camera Stream, Google Gemini Vision API (optional AI mode)
- **Computer Vision Pipeline:**
  - Color-space conversion (RGB to HSV)
  - Perimeter background profiling & adaptive thresholding
  - Morphological Closing & Opening filters
  - 8-Connected Component Analysis & flood-fill hole closure
  - Convex Hull & Ramer-Douglas-Peucker polygonal simplification
  - Triangle-to-Contour IoU (Intersection over Union) verification
  - Curvature & vertex salience estimation

---

### Implementation

#### For Software:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/jabinnida/useless_project_amigos.git
   cd useless_project_amigos
   ```

2. **Run:**
   - Simply double-click and open `index.html` in any modern web browser (Google Chrome, Microsoft Edge, Firefox, Brave).
   - *Optional local server:* Run with PowerShell:
     ```powershell
     powershell -ExecutionPolicy Bypass -File serve.ps1
     ```
     Then navigate to `http://localhost:8000`.

---

### Project Documentation

#### Screenshots
*(Add screenshots of your application in action below)*
- **Upload & Camera Interface:** Drop a photo or launch the live camera feed with mirror and flip support.
- **Analysis View:** Original photo with background intact, overlaid with Apex (red), Bottom-Left (green), and Bottom-Right (blue) markers, neon triangle, and angle displays.
- **Score Breakdown:** Accuracy percentage ring, side length differences, equilateral deviation, and verdict.

---

### Project Demo

#### Video
- [https://drive.google.com/file/d/1jiRYFvcaEzxk1uwBkBJ00ubRpM13cYSh/view?usp=drivesdk]
- *Short demonstration showing photo upload, live camera capture, automatic corner detection, and non-samosa rejection.*

---

## Team Contributions
- **Team Lead:** Project architecture, computer vision segmentation pipeline, geometric calculation engine, and camera integration.
- **Member 2:** UI/UX styling, glassmorphism design, non-samosa validation heuristics, testing, and documentation.

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)
