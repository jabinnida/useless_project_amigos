/**
 * Samosa Triangle Accuracy Finder — app.js (v4 — AI Assisted + CV Refinement)
 *
 * PIPELINE:
 *  1. AI Samosa Detection (Gemini Vision API):
 *     - Isolate samosa from hand / background
 *     - Return bounding box + normalized 3 corner coordinates (apex, bottomLeft, bottomRight)
 *     - No angle guessing from AI! Strictly coordinate detection.
 *  2. CV Corner Refinement:
 *     - Build localized contour along the samosa silhouette
 *     - Local curvature & exterior-angle search around each AI-predicted corner
 *     - Snap corners onto true outermost silhouette boundary
 *     - Reject background & hand points
 *  3. Mathematical Geometry Engine:
 *     - Side lengths: AB, BC, CA (Euclidean distance)
 *     - Interior angles: Law of Cosines (A, B, C)
 *     - Verification: A + B + C ≈ 180°
 *     - Shoelace triangle area vs segmented samosa area
 *     - Contour-to-triangle similarity & IoU
 *     - Triangle Accuracy Score (0–100%)
 *  4. Precision Mode:
 *     - Interactive manual 3-point placement on canvas
 *     - Direct calculation with identical mathematical geometry
 *  5. Debug Mode:
 *     - Shows AI region/box, raw AI corners, refined corners, contour & triangle
 */

'use strict';

/* ═══════════════════════════════════════════════════════
   0. CONFIGURATION & STATE
   ═══════════════════════════════════════════════════════ */
const CONFIG = {
  maxProcessDim: 720,      // resize long edge for fast CV & AI inference
  geminiModel: 'gemini-2.5-flash',
  fallbackModel: 'gemini-2.0-flash',
  confidenceThreshold: 0.55,
  localSearchRadiusPct: 0.15 // search radius for corner refinement (fraction of samosa size)
};

const STATE = {
  apiKey: localStorage.getItem('gemini_api_key') || '',
  mode: 'ai',              // 'ai' | 'precision'
  debugMode: false,
  rawImage: null,          // Image object
  scaledCanvas: null,      // off-screen canvas with resized image
  scaleRatio: 1.0,         // rawImage dimensions / scaledCanvas dimensions
  lastResult: null,        // caches last analysis output for fast re-renders
  precisionCorners: [],    // [{x, y}] on display canvas coords
  analyzing: false
};

/* ═══════════════════════════════════════════════════════
   1. MATHEMATICAL GEOMETRY ENGINE
   ═══════════════════════════════════════════════════════ */

/** Euclidean distance between two points */
function dist(p1, p2) {
  return Math.hypot(p2.x - p1.x, p2.y - p1.y);
}

/**
 * Calculate interior angles from 3 vertices strictly via Law of Cosines.
 * Returns angles in degrees for A (apex), B (bottom-left), C (bottom-right).
 */
function computeAnglesFromVertices(A, B, C) {
  const a = dist(B, C); // opposite to A
  const b = dist(A, C); // opposite to B
  const c = dist(A, B); // opposite to C

  if (a <= 0 || b <= 0 || c <= 0) {
    return { A: 60, B: 60, C: 60, sum: 180, a, b, c, valid: false };
  }

  // Clamp cosine values to [-1, 1] to avoid NaN from numerical float inaccuracies
  const clampCos = val => Math.max(-1, Math.min(1, val));

  const cosA = clampCos((b * b + c * c - a * a) / (2 * b * c));
  const cosB = clampCos((a * a + c * c - b * b) / (2 * a * c));
  const cosC = clampCos((a * a + b * b - c * c) / (2 * a * b));

  const radToDeg = rad => (rad * 180) / Math.PI;

  const angA = radToDeg(Math.acos(cosA));
  const angB = radToDeg(Math.acos(cosB));
  const angC = radToDeg(Math.acos(cosC));
  const sum = angA + angB + angC;

  return {
    A: angA,
    B: angB,
    C: angC,
    sum: sum,
    sideA: a, // BC
    sideB: b, // AC
    sideC: c, // AB
    valid: Math.abs(sum - 180) < 1.0 && angA > 5 && angB > 5 && angC > 5
  };
}

/** Shoelace formula for triangle area from vertices */
function triangleArea(A, B, C) {
  return Math.abs(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2;
}

/** Polygon area from arbitrary vertices list */
function polygonArea(pts) {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i].x * pts[j].y;
    area -= pts[j].x * pts[i].y;
  }
  return Math.abs(area) / 2;
}

/**
 * Compute full geometry & accuracy score comparing samosa silhouette with triangle
 */
function computeGeometryResults(triangle, samosaMask, hullPts, W, H) {
  const [A, B, C] = triangle;
  const geom = computeAnglesFromVertices(A, B, C);
  const triAreaVal = triangleArea(A, B, C);

  // Measure samosa silhouette area
  let contourArea = 0;
  if (samosaMask) {
    for (let i = 0; i < samosaMask.length; i++) {
      if (samosaMask[i]) contourArea++;
    }
  } else if (hullPts && hullPts.length >= 3) {
    contourArea = polygonArea(hullPts);
  } else {
    contourArea = triAreaVal;
  }

  // Rasterize triangle to calculate IoU against samosa mask
  let triMaskCount = 0;
  let intersection = 0;
  let union = 0;

  if (samosaMask && W > 0 && H > 0) {
    const triMask = rasterizeTriangle(A, B, C, W, H);
    for (let i = 0; i < W * H; i++) {
      const isTri = triMask[i] > 0;
      const isSam = samosaMask[i] > 0;
      if (isTri) triMaskCount++;
      if (isTri && isSam) intersection++;
      if (isTri || isSam) union++;
    }
  }

  const iou = union > 0 ? (intersection / union) * 100 : 70;
  const areaRatio = contourArea > 0 ? Math.min(triAreaVal / contourArea, contourArea / triAreaVal) * 100 : 70;

  // Measure average distance from hull boundary to nearest triangle edge
  let edgeFitScore = 75;
  if (hullPts && hullPts.length > 0) {
    let totalDist = 0;
    for (const p of hullPts) {
      const d1 = pointToSegmentDist(p, A, B);
      const d2 = pointToSegmentDist(p, B, C);
      const d3 = pointToSegmentDist(p, C, A);
      totalDist += Math.min(d1, d2, d3);
    }
    const avgDist = totalDist / hullPts.length;
    const diag = Math.hypot(W, H);
    edgeFitScore = Math.max(0, 100 - (avgDist / (diag * 0.04)) * 100);
  }

  // Equilateral balance score (how close to 60-60-60 is it?)
  const angleDev = (Math.abs(geom.A - 60) + Math.abs(geom.B - 60) + Math.abs(geom.C - 60)) / 3;
  const symmetryScore = Math.max(0, 100 - angleDev * 1.6);

  // Overall Triangle Accuracy Score
  // Weighted: IoU (40%) + Area Match (30%) + Edge Fit (20%) + Triangle Symmetry (10%)
  let accuracyScore = Math.round(iou * 0.40 + areaRatio * 0.30 + edgeFitScore * 0.20 + symmetryScore * 0.10);
  accuracyScore = Math.max(12, Math.min(99, accuracyScore));

  return {
    vertices: { apex: A, bottomLeft: B, bottomRight: C },
    angles: { A: geom.A, B: geom.B, C: geom.C, sum: geom.sum },
    sideLengths: { AB: geom.sideC, BC: geom.sideA, CA: geom.sideB },
    triArea: triAreaVal,
    contourArea: contourArea,
    areaRatio: areaRatio,
    iou: iou,
    edgeFit: edgeFitScore,
    symmetryScore: symmetryScore,
    accuracyScore: accuracyScore
  };
}

/** Distance from point P to line segment VW */
function pointToSegmentDist(p, v, w) {
  const l2 = (w.x - v.x) ** 2 + (w.y - v.y) ** 2;
  if (l2 === 0) return dist(p, v);
  let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
  t = Math.max(0, Math.min(1, t));
  const proj = { x: v.x + t * (w.x - v.x), y: v.y + t * (w.y - v.y) };
  return dist(p, proj);
}

/** Rasterize triangle into binary Uint8Array mask */
function rasterizeTriangle(p1, p2, p3, w, h) {
  const mask = new Uint8Array(w * h);
  const minX = Math.max(0, Math.floor(Math.min(p1.x, p2.x, p3.x)));
  const maxX = Math.min(w - 1, Math.ceil(Math.max(p1.x, p2.x, p3.x)));
  const minY = Math.max(0, Math.floor(Math.min(p1.y, p2.y, p3.y)));
  const maxY = Math.min(h - 1, Math.ceil(Math.max(p1.y, p2.y, p3.y)));

  const sign = (p, a, b) => (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const pt = { x: x + 0.5, y: y + 0.5 };
      const d1 = sign(pt, p1, p2);
      const d2 = sign(pt, p2, p3);
      const d3 = sign(pt, p3, p1);
      const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
      const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
      if (!(hasNeg && hasPos)) {
        mask[y * w + x] = 255;
      }
    }
  }
  return mask;
}

/* ═══════════════════════════════════════════════════════
   2. GEMINI VISION AI DETECTOR
   ═══════════════════════════════════════════════════════ */

/**
 * Calls Gemini Vision API to detect samosa and locate 3 corner points.
 * Explicitly forbids guessing angles — only returns object bbox and corner coords.
 */
async function callGeminiVision(canvas) {
  if (!STATE.apiKey) {
    throw new Error('NO_API_KEY');
  }

  // Convert canvas to JPEG base64 (stripped of data URL header)
  const base64Data = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];

  const prompt = `You are a high-precision computer vision model. Your task is to detect the main samosa in the image and locate its 3 main corners.
CRITICAL INSTRUCTION:
Do NOT calculate, guess, or output any angles or accuracy scores.
You must ONLY locate the 2D pixel coordinates and bounding box. The application calculates all angles mathematically.

DETECTION GUIDELINES:
1. Isolate the main samosa from the background.
2. If someone is holding the samosa with their hand/fingers, completely exclude the hand, fingers, and background from the samosa vertices.
3. Identify the 3 physical corners of the samosa triangle:
   - "apex": The top tip/peak of the samosa.
   - "bottomLeft": The bottom-left corner of the samosa.
   - "bottomRight": The bottom-right corner of the samosa.
4. Return normalized integer coordinates on a 0 to 1000 scale (where (0,0) is top-left and (1000,1000) is bottom-right).

OUTPUT FORMAT:
Output MUST be strictly valid JSON adhering to this exact format:
{
  "hasSamosa": true,
  "confidence": 0.92,
  "boundingBox": { "ymin": 50, "xmin": 200, "ymax": 950, "xmax": 750 },
  "corners": {
    "apex": { "x": 480, "y": 80 },
    "bottomLeft": { "x": 220, "y": 820 },
    "bottomRight": { "x": 720, "y": 780 }
  },
  "notes": "Samosa detected with hand on left side excluded."
}`;

  const requestBody = {
    contents: [
      {
        parts: [
          { text: prompt },
          {
            inline_data: {
              mime_type: 'image/jpeg',
              data: base64Data
            }
          }
        ]
      }
    ],
    generationConfig: {
      temperature: 0.1,
      response_mime_type: 'application/json'
    }
  };

  const modelsToTry = [CONFIG.geminiModel, CONFIG.fallbackModel];
  let lastError = null;

  for (const model of modelsToTry) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(STATE.apiKey)}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const errMsg = errData.error?.message || `HTTP ${response.status} ${response.statusText}`;
        throw new Error(errMsg);
      }

      const resJson = await response.json();
      const rawText = resJson.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!rawText) throw new Error('No content returned from Gemini Vision API.');

      // Parse JSON from response
      const cleanJson = rawText.replace(/```json/gi, '').replace(/```/g, '').trim();
      const result = JSON.parse(cleanJson);

      if (!result.hasSamosa || !result.corners) {
        throw new Error('No samosa recognized by the AI in this image.');
      }

      return result;
    } catch (err) {
      console.warn(`[AI] Attempt with ${model} failed:`, err.message);
      lastError = err;
    }
  }

  throw lastError || new Error('Failed to reach Gemini Vision API.');
}

/* ═══════════════════════════════════════════════════════
   3. COMPUTER VISION CORNER REFINEMENT
   ═══════════════════════════════════════════════════════ */

/**
 * Builds localized edge & silhouette contour inside the AI bounding box
 * and refines corner positions to the strongest geometric outer boundary.
 */
function refineCornersWithContour(aiResult, imgData, W, H) {
  const bboxNorm = aiResult.boundingBox || { xmin: 50, ymin: 50, xmax: 950, ymax: 950 };
  const bbox = {
    xmin: Math.max(0, Math.floor((bboxNorm.xmin / 1000) * W)),
    ymin: Math.max(0, Math.floor((bboxNorm.ymin / 1000) * H)),
    xmax: Math.min(W - 1, Math.ceil((bboxNorm.xmax / 1000) * W)),
    ymax: Math.min(H - 1, Math.ceil((bboxNorm.ymax / 1000) * H))
  };

  const rawCornersNorm = aiResult.corners;
  const rawApex = { x: (rawCornersNorm.apex.x / 1000) * W, y: (rawCornersNorm.apex.y / 1000) * H };
  const rawBL = { x: (rawCornersNorm.bottomLeft.x / 1000) * W, y: (rawCornersNorm.bottomLeft.y / 1000) * H };
  const rawBR = { x: (rawCornersNorm.bottomRight.x / 1000) * W, y: (rawCornersNorm.bottomRight.y / 1000) * H };

  // Generate segmentation mask restricted to bounding box
  const { mask, hullPts } = segmentSamosaRegion(imgData, W, H, bbox);

  if (!hullPts || hullPts.length < 3) {
    // If hull extraction within bbox is degenerate, use raw AI corners
    return {
      refinedCorners: [rawApex, rawBL, rawBR],
      rawAiCorners: [rawApex, rawBL, rawBR],
      hullPts: [rawApex, rawBL, rawBR],
      mask: mask,
      bbox: bbox
    };
  }

  // Refine each of the 3 corners by searching locally along the hull/contour
  const samosaSize = Math.max(bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin);
  const searchRadius = Math.max(16, samosaSize * CONFIG.localSearchRadiusPct);

  const refinedApex = findBestCornerPoint(rawApex, hullPts, searchRadius, 'top');
  const refinedBL = findBestCornerPoint(rawBL, hullPts, searchRadius, 'bottomLeft');
  const refinedBR = findBestCornerPoint(rawBR, hullPts, searchRadius, 'bottomRight');

  return {
    refinedCorners: [refinedApex, refinedBL, refinedBR],
    rawAiCorners: [rawApex, rawBL, rawBR],
    hullPts: hullPts,
    mask: mask,
    bbox: bbox
  };
}

/**
 * Searches near targetPt on hullPts to find the point with strongest curvature / outer prominence.
 */
function findBestCornerPoint(targetPt, hullPts, maxRadius, role) {
  let bestPt = targetPt;
  let bestScore = -Infinity;
  const n = hullPts.length;

  for (let i = 0; i < n; i++) {
    const pt = hullPts[i];
    const d = dist(pt, targetPt);
    if (d > maxRadius) continue;

    // Geometric curvature: angle between incoming and outgoing edges along hull
    const prev = hullPts[(i - 2 + n) % n];
    const next = hullPts[(i + 2) % n];

    const v1 = { x: prev.x - pt.x, y: prev.y - pt.y };
    const v2 = { x: next.x - pt.x, y: next.y - pt.y };
    const len1 = Math.hypot(v1.x, v1.y) || 1;
    const len2 = Math.hypot(v2.x, v2.y) || 1;

    // Cosine of interior turn angle (sharper corner -> smaller/negative cos or acute turn)
    const dot = (v1.x * v2.x + v1.y * v2.y) / (len1 * len2);
    const curvature = 1.0 - dot; // range 0..2 (higher = sharper vertex)

    // Role-based directional bonus
    let dirBonus = 0;
    if (role === 'top') {
      dirBonus = -pt.y * 0.05; // lower Y = higher in image
    } else if (role === 'bottomLeft') {
      dirBonus = -pt.x * 0.03 + pt.y * 0.03;
    } else if (role === 'bottomRight') {
      dirBonus = pt.x * 0.03 + pt.y * 0.03;
    }

    // Distance penalty (prefer staying reasonably close to AI guidance)
    const distPenalty = (d / maxRadius) * 0.8;

    const totalScore = curvature + dirBonus - distPenalty;
    if (totalScore > bestScore) {
      bestScore = totalScore;
      bestPt = pt;
    }
  }

  return bestPt;
}

/**
 * Segment the samosa region bounded inside bbox, rejecting background & hand
 */
function segmentSamosaRegion(imgData, W, H, bbox) {
  const data = imgData.data;
  const mask = new Uint8Array(W * H);

  // Sample colors from bbox to compute golden samosa crust threshold
  const samosaPixels = [];
  for (let y = bbox.ymin; y <= bbox.ymax; y++) {
    for (let x = bbox.xmin; x <= bbox.xmax; x++) {
      const idx = (y * W + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      // Golden/fried pastry test: R > B, warm hue, not overly dark or blown out
      const isPastry = r > 70 && g > 40 && (r > b + 15) && (r + g + b < 720);
      if (isPastry) {
        mask[y * W + x] = 255;
        samosaPixels.push({ x, y });
      }
    }
  }

  // Morphological close to bridge chutney gaps & filling edges
  const cleanedMask = morphClose(mask, W, H, 3);

  // Extract boundary contour points
  const boundaryPts = [];
  for (let y = bbox.ymin + 1; y < bbox.ymax - 1; y++) {
    for (let x = bbox.xmin + 1; x < bbox.xmax - 1; x++) {
      if (cleanedMask[y * W + x] === 255) {
        // Check if 4-neighbor is 0
        if (
          cleanedMask[(y - 1) * W + x] === 0 ||
          cleanedMask[(y + 1) * W + x] === 0 ||
          cleanedMask[y * W + (x - 1)] === 0 ||
          cleanedMask[y * W + (x + 1)] === 0
        ) {
          boundaryPts.push({ x, y });
        }
      }
    }
  }

  if (boundaryPts.length < 6) {
    return { mask: cleanedMask, hullPts: null };
  }

  const hull = convexHull(boundaryPts);
  return { mask: cleanedMask, hullPts: hull };
}

/** 3x3 Morphological Close (Dilation followed by Erosion) */
function morphClose(src, w, h, radius = 2) {
  const dilated = new Uint8Array(w * h);
  const result = new Uint8Array(w * h);

  // Dilate
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let maxVal = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -radius; dy <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          if (src[ny * w + nx] > maxVal) maxVal = 255;
        }
      }
      dilated[y * w + x] = maxVal;
    }
  }

  // Erode
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let minVal = 255;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          if (dilated[ny * w + nx] === 0) { minVal = 0; break; }
        }
        if (minVal === 0) break;
      }
      result[y * w + x] = minVal;
    }
  }

  return result;
}

/** Convex Hull via Monotone Chain Algorithm */
function convexHull(pts) {
  if (pts.length <= 3) return pts.slice();
  const sorted = pts.slice().sort((a, b) => a.x !== b.x ? a.x - b.x : a.y - b.y);

  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

/* ═══════════════════════════════════════════════════════
   4. PURE-CV OFFLINE FALLBACK PIPELINE
   ═══════════════════════════════════════════════════════ */

/** Runs offline pure-CV when no API key is available */
function runPureCvFallback(canvas) {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, W, H);
  const data = imgData.data;

  // LAB background subtraction mask
  const mask = new Uint8Array(W * H);
  const pts = [];

  // Foreground mask via warm pastry / golden hue test
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const isPastry = (r > 80 && g > 50 && r > b + 15) && (x > W * 0.05 && x < W * 0.95 && y > H * 0.05 && y < H * 0.95);
      if (isPastry) {
        mask[y * W + x] = 255;
        pts.push({ x, y });
      }
    }
  }

  if (pts.length < 20) {
    // Default triangular estimate
    const defaultApex = { x: W * 0.5, y: H * 0.15 };
    const defaultBL = { x: W * 0.2, y: H * 0.85 };
    const defaultBR = { x: W * 0.8, y: H * 0.85 };
    return {
      refinedCorners: [defaultApex, defaultBL, defaultBR],
      rawAiCorners: null,
      hullPts: [defaultApex, defaultBL, defaultBR],
      mask: mask,
      confidence: 0.40
    };
  }

  const hull = convexHull(pts);

  // Find 3 extremal points: min Y (apex), min X with high Y (BL), max X with high Y (BR)
  let apex = hull[0], bl = hull[0], br = hull[0];
  let minApexY = Infinity;
  let bestBLScore = -Infinity;
  let bestBRScore = -Infinity;

  for (const p of hull) {
    if (p.y < minApexY) {
      minApexY = p.y;
      apex = p;
    }
    const blScore = -p.x * 1.5 + p.y;
    if (blScore > bestBLScore) {
      bestBLScore = blScore;
      bl = p;
    }
    const brScore = p.x * 1.5 + p.y;
    if (brScore > bestBRScore) {
      bestBRScore = brScore;
      br = p;
    }
  }

  return {
    refinedCorners: [apex, bl, br],
    rawAiCorners: null,
    hullPts: hull,
    mask: mask,
    confidence: 0.50
  };
}

/* ═══════════════════════════════════════════════════════
   5. OVERLAY RENDERING & DEBUG MODE
   ═══════════════════════════════════════════════════════ */

/**
 * Draws the analyzed samosa image with triangle overlay, angle labels, and debug layers
 */
function renderOverlay(canvas, result, showDebug) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // Clear & draw base image
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(STATE.rawImage, 0, 0, W, H);

  const [A, B, C] = result.refinedCorners;
  const angles = result.geometry.angles;

  // 1. Debug layers if active
  if (showDebug) {
    // Samosa bounding box (dashed blue)
    if (result.bbox) {
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.75)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.strokeRect(result.bbox.xmin, result.bbox.ymin, result.bbox.xmax - result.bbox.xmin, result.bbox.ymax - result.bbox.ymin);
      ctx.fillStyle = 'rgba(0, 212, 255, 0.9)';
      ctx.font = 'bold 11px Outfit, sans-serif';
      ctx.fillText('AI SAMOSA REGION', result.bbox.xmin + 6, result.bbox.ymin + 15);
      ctx.restore();
    }

    // Outer silhouette contour (green)
    if (result.hullPts && result.hullPts.length > 2) {
      ctx.save();
      ctx.strokeStyle = 'rgba(57, 217, 138, 0.7)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(result.hullPts[0].x, result.hullPts[0].y);
      for (let i = 1; i < result.hullPts.length; i++) {
        ctx.lineTo(result.hullPts[i].x, result.hullPts[i].y);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }

    // Raw AI predicted corners (magenta rings)
    if (result.rawAiCorners) {
      const labels = ['Raw Apex', 'Raw BL', 'Raw BR'];
      result.rawAiCorners.forEach((pt, i) => {
        ctx.save();
        ctx.strokeStyle = '#ff00ea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = '#ff00ea';
        ctx.font = 'bold 11px Outfit, sans-serif';
        ctx.fillText(labels[i], pt.x + 12, pt.y - 4);
        ctx.restore();
      });
    }
  }

  // 2. Final Triangle Geometry Overlay
  ctx.save();
  // Semi-transparent triangle fill
  ctx.fillStyle = 'rgba(255, 107, 43, 0.18)';
  ctx.beginPath();
  ctx.moveTo(A.x, A.y);
  ctx.lineTo(B.x, B.y);
  ctx.lineTo(C.x, C.y);
  ctx.closePath();
  ctx.fill();

  // Glowing triangle perimeter edges
  ctx.strokeStyle = '#ff6b2b';
  ctx.lineWidth = 3.5;
  ctx.shadowColor = '#ff6b2b';
  ctx.shadowBlur = 12;
  ctx.stroke();
  ctx.restore();

  // 3. Vertices & Angle Badges
  const vertices = [
    { pt: A, name: 'A (Apex)', angle: angles.A },
    { pt: B, name: 'B (BL)', angle: angles.B },
    { pt: C, name: 'C (BR)', angle: angles.C }
  ];

  vertices.forEach(v => {
    // Outer vertex glow
    ctx.save();
    ctx.fillStyle = '#ffd700';
    ctx.shadowColor = '#ffd700';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(v.pt.x, v.pt.y, 6.5, 0, Math.PI * 2);
    ctx.fill();

    // Center core
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(v.pt.x, v.pt.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Angle & Vertex pill label
    drawAnglePill(ctx, v.pt.x, v.pt.y, `${v.name.slice(0, 1)}: ${v.angle.toFixed(1)}°`);
  });
}

/** Draws rounded glass pill for angle measurement on the canvas */
function drawAnglePill(ctx, x, y, text) {
  ctx.save();
  ctx.font = 'bold 12px Outfit, sans-serif';
  const pad = 6;
  const tw = ctx.measureText(text).width;
  const pw = tw + pad * 2;
  const ph = 22;
  const px = x + 10;
  const py = y - 10;

  ctx.fillStyle = 'rgba(10, 10, 20, 0.88)';
  ctx.strokeStyle = 'rgba(255, 215, 0, 0.7)';
  ctx.lineWidth = 1.2;

  // Rounded rectangle
  ctx.beginPath();
  ctx.roundRect(px, py - ph + 4, pw, ph, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#ffd700';
  ctx.fillText(text, px + pad, py);
  ctx.restore();
}

/* ═══════════════════════════════════════════════════════
   6. UI RENDERING & MEASUREMENTS DISPLAY
   ═══════════════════════════════════════════════════════ */

function renderMeasurements(geom) {
  const grid = document.getElementById('measurement-grid');
  grid.innerHTML = `
    <div class="m-item">
      <div class="m-label">Interior Angles</div>
      <div class="m-value highlight">${geom.angles.A.toFixed(1)}° · ${geom.angles.B.toFixed(1)}° · ${geom.angles.C.toFixed(1)}°</div>
    </div>
    <div class="m-item">
      <div class="m-label">Angle Sum (180° Check)</div>
      <div class="m-value">${geom.angles.sum.toFixed(1)}° ${Math.abs(geom.angles.sum - 180) < 0.5 ? '✓' : ''}</div>
    </div>
    <div class="m-item">
      <div class="m-label">Side Lengths</div>
      <div class="m-value">${Math.round(geom.sideLengths.AB)}px · ${Math.round(geom.sideLengths.BC)}px · ${Math.round(geom.sideLengths.CA)}px</div>
    </div>
    <div class="m-item">
      <div class="m-label">Shape IoU Match</div>
      <div class="m-value">${Math.round(geom.iou)}%</div>
    </div>
    <div class="m-item">
      <div class="m-label">Triangle Area</div>
      <div class="m-value">${Math.round(geom.triArea).toLocaleString()} px²</div>
    </div>
    <div class="m-item">
      <div class="m-label">Samosa Area</div>
      <div class="m-value">${Math.round(geom.contourArea).toLocaleString()} px²</div>
    </div>
  `;

  // Vertices Table
  const tbody = document.getElementById('vertices-body');
  tbody.innerHTML = `
    <tr><td><strong>A (Apex)</strong></td><td>${Math.round(geom.vertices.apex.x)}</td><td>${Math.round(geom.vertices.apex.y)}</td></tr>
    <tr><td><strong>B (Bottom-Left)</strong></td><td>${Math.round(geom.vertices.bottomLeft.x)}</td><td>${Math.round(geom.vertices.bottomLeft.y)}</td></tr>
    <tr><td><strong>C (Bottom-Right)</strong></td><td>${Math.round(geom.vertices.bottomRight.x)}</td><td>${Math.round(geom.vertices.bottomRight.y)}</td></tr>
  `;
}

function renderScore(score) {
  const numEl = document.getElementById('score-number');
  const labelEl = document.getElementById('result-label');
  const ringFill = document.getElementById('ring-fill');

  // Animated number counter
  let current = 0;
  const step = Math.max(1, Math.round(score / 30));
  const timer = setInterval(() => {
    current = Math.min(score, current + step);
    numEl.textContent = current;
    if (current >= score) clearInterval(timer);
  }, 25);

  // Animated circle ring (circumference = 2 * PI * 82 ≈ 515.22)
  const total = 515.22;
  const offset = total - (score / 100) * total;
  ringFill.style.strokeDashoffset = offset;

  // Fun rating tiers
  let msg = '';
  if (score >= 90) {
    msg = '🔺 Perfect Triangle Samosa 🔺';
  } else if (score >= 75) {
    msg = '😎 Pretty Triangular 😎';
  } else if (score >= 50) {
    msg = '🤔 Samosa-ish 🤔';
  } else {
    msg = "💀 Bro, that's not a triangle 💀";
  }
  labelEl.textContent = msg;
}

function updateConfidenceUI(confidence, isAiMode) {
  const confBlock = document.getElementById('confidence-block');
  const confPct = document.getElementById('confidence-pct');
  const confBar = document.getElementById('confidence-bar-fill');
  const confNote = document.getElementById('confidence-note');

  if (!isAiMode || confidence === null || confidence === undefined) {
    confBlock.classList.add('hidden');
    return;
  }

  confBlock.classList.remove('hidden');
  const pct = Math.round(confidence * 100);
  confPct.textContent = `${pct}%`;
  confBar.style.width = `${pct}%`;

  if (confidence < CONFIG.confidenceThreshold) {
    confNote.classList.remove('hidden');
  } else {
    confNote.classList.add('hidden');
  }
}

function updateDetectionBadge(modeType) {
  const badge = document.getElementById('detection-badge');
  badge.className = 'detection-badge';
  if (modeType === 'ai') {
    badge.classList.add('badge-ai');
    badge.textContent = '🤖 AI Refined';
  } else if (modeType === 'precision') {
    badge.classList.add('badge-precision');
    badge.textContent = '✋ Precision Mode';
  } else {
    badge.classList.add('badge-cv');
    badge.textContent = '⚡ Pure CV Mode';
  }
}

/* ═══════════════════════════════════════════════════════
   7. MAIN ANALYSIS DISPATCHER
   ═══════════════════════════════════════════════════════ */

async function startAnalysis(img) {
  STATE.rawImage = img;
  showLoading();
  resetSteps();

  try {
    // Step 1: Prepare scaled canvas for fast processing & inference
    activateStep(0);
    await sleep(40);

    const maxDim = CONFIG.maxProcessDim;
    let W = img.naturalWidth || img.width;
    let H = img.naturalHeight || img.height;
    const ratio = Math.min(1.0, maxDim / Math.max(W, H));
    const targetW = Math.round(W * ratio);
    const targetH = Math.round(H * ratio);

    const procCanvas = document.getElementById('proc-canvas');
    procCanvas.width = targetW;
    procCanvas.height = targetH;
    const pctx = procCanvas.getContext('2d');
    pctx.drawImage(img, 0, 0, targetW, targetH);
    STATE.scaledCanvas = procCanvas;
    STATE.scaleRatio = 1 / ratio;

    // Display Canvas setup (matches scaled resolution for high performance)
    const resultCanvas = document.getElementById('result-canvas');
    resultCanvas.width = targetW;
    resultCanvas.height = targetH;

    let analysisResult = null;

    if (STATE.mode === 'precision') {
      // Precision Mode: allow user to click corners
      initPrecisionMode(resultCanvas);
      return;
    }

    // Step 2: AI Samosa Detection
    activateStep(1);
    await sleep(40);

    let aiResult = null;
    let usedCvOnly = false;

    if (STATE.apiKey) {
      try {
        aiResult = await callGeminiVision(procCanvas);
      } catch (aiErr) {
        console.warn('[AI] Vision API failed, falling back to pure CV:', aiErr.message);
        usedCvOnly = true;
      }
    } else {
      console.log('[AI] No API key configured. Using offline CV pipeline.');
      usedCvOnly = true;
    }

    // Step 3: Corner Refinement
    activateStep(2);
    await sleep(40);

    if (aiResult && !usedCvOnly) {
      const imgData = pctx.getImageData(0, 0, targetW, targetH);
      const refined = refineCornersWithContour(aiResult, imgData, targetW, targetH);
      const geom = computeGeometryResults(refined.refinedCorners, refined.mask, refined.hullPts, targetW, targetH);

      analysisResult = {
        refinedCorners: refined.refinedCorners,
        rawAiCorners: refined.rawAiCorners,
        hullPts: refined.hullPts,
        bbox: refined.bbox,
        mask: refined.mask,
        geometry: geom,
        confidence: aiResult.confidence ?? 0.88,
        modeType: 'ai'
      };
    } else {
      // Pure CV fallback
      const cvRes = runPureCvFallback(procCanvas);
      const geom = computeGeometryResults(cvRes.refinedCorners, cvRes.mask, cvRes.hullPts, targetW, targetH);

      analysisResult = {
        refinedCorners: cvRes.refinedCorners,
        rawAiCorners: null,
        hullPts: cvRes.hullPts,
        bbox: null,
        mask: cvRes.mask,
        geometry: geom,
        confidence: cvRes.confidence,
        modeType: 'cv'
      };
    }

    // Step 4: Calculate Score & Render
    activateStep(3);
    await sleep(40);

    STATE.lastResult = analysisResult;
    renderOverlay(resultCanvas, analysisResult, STATE.debugMode);
    renderScore(analysisResult.geometry.accuracyScore);
    renderMeasurements(analysisResult.geometry);
    updateConfidenceUI(analysisResult.confidence, analysisResult.modeType === 'ai');
    updateDetectionBadge(analysisResult.modeType);

    showResults();

  } catch (err) {
    console.error('Analysis error:', err);
    showError('Analysis failed: ' + (err.message || String(err)));
  }
}

/* ═══════════════════════════════════════════════════════
   8. PRECISION MODE (MANUAL CORNER SELECTION)
   ═══════════════════════════════════════════════════════ */

const CORNER_NAMES = ['apex (top tip)', 'bottom-left corner', 'bottom-right corner'];

function initPrecisionMode(canvas) {
  STATE.precisionCorners = [];
  showResults();

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(STATE.rawImage, 0, 0, canvas.width, canvas.height);

  updateDetectionBadge('precision');
  updateConfidenceUI(null, false);

  const overlay = document.getElementById('precision-overlay');
  const controls = document.getElementById('precision-controls');
  overlay.classList.remove('hidden');
  controls.classList.remove('hidden');

  updatePrecisionUI();
}

function updatePrecisionUI() {
  const overlay = document.getElementById('precision-overlay');
  const hint = document.getElementById('precision-hint');
  const cornerName = document.getElementById('corner-name');
  const countEl = document.getElementById('prec-count');
  const calcBtn = document.getElementById('precision-calculate');

  const count = STATE.precisionCorners.length;
  countEl.textContent = `${count} / 3 corners placed`;

  // Update dots
  for (let i = 0; i < 3; i++) {
    const dot = document.getElementById(`prec-dot-${i}`);
    if (i < count) dot.classList.add('active');
    else dot.classList.remove('active');
  }

  // Remove existing corner pins
  overlay.querySelectorAll('.corner-pin').forEach(pin => pin.remove());

  // Render corner pins
  const canvas = document.getElementById('result-canvas');
  const rect = canvas.getBoundingClientRect();
  const scaleX = rect.width / canvas.width;
  const scaleY = rect.height / canvas.height;

  STATE.precisionCorners.forEach((pt, i) => {
    const pin = document.createElement('div');
    pin.className = 'corner-pin';
    pin.textContent = i + 1;
    pin.style.left = `${pt.x * scaleX}px`;
    pin.style.top = `${pt.y * scaleY}px`;
    overlay.appendChild(pin);
  });

  if (count < 3) {
    cornerName.textContent = CORNER_NAMES[count];
    hint.classList.remove('hidden');
    calcBtn.classList.add('hidden');
  } else {
    hint.classList.add('hidden');
    calcBtn.classList.remove('hidden');
    // Auto-calculate score once 3 points are placed!
    calculatePrecisionResult();
  }
}

function calculatePrecisionResult() {
  if (STATE.precisionCorners.length !== 3) return;

  const canvas = document.getElementById('result-canvas');
  const W = canvas.width, H = canvas.height;
  const [A, B, C] = STATE.precisionCorners;

  const geom = computeGeometryResults([A, B, C], null, [A, B, C], W, H);

  const analysisResult = {
    refinedCorners: [A, B, C],
    rawAiCorners: null,
    hullPts: [A, B, C],
    bbox: null,
    mask: null,
    geometry: geom,
    confidence: 1.0,
    modeType: 'precision'
  };

  STATE.lastResult = analysisResult;
  renderOverlay(canvas, analysisResult, STATE.debugMode);
  renderScore(geom.accuracyScore);
  renderMeasurements(geom);
  updateDetectionBadge('precision');
}

/* ═══════════════════════════════════════════════════════
   9. DOM WIRING & EVENT HANDLERS
   ═══════════════════════════════════════════════════════ */

// Elements
const uploadZone     = document.getElementById('upload-zone');
const fileInput      = document.getElementById('file-input');
const cameraBtn      = document.getElementById('camera-btn');
const cameraInput    = document.getElementById('camera-input');
const sampleBtn      = document.getElementById('sample-btn');
const uploadSection  = document.getElementById('upload-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const retryBtn       = document.getElementById('retry-btn');
const errorToast     = document.getElementById('error-toast');
const errorMsg       = document.getElementById('error-message');
const steps          = [1, 2, 3, 4].map(i => document.getElementById('step-' + i));

// Settings & API Modal Elements
const settingsBtn     = document.getElementById('settings-btn');
const apiModal        = document.getElementById('api-modal-backdrop');
const apiCloseBtn     = document.getElementById('api-modal-close');
const apiKeyInput     = document.getElementById('api-key-input');
const apiKeyToggle    = document.getElementById('api-key-toggle');
const apiKeySaveBtn   = document.getElementById('api-key-save');
const apiKeyCancelBtn = document.getElementById('api-key-cancel');
const apiKeyClearBtn  = document.getElementById('api-key-clear');
const apiKeyStatus    = document.getElementById('api-key-status');
const noKeyNudge      = document.getElementById('no-key-nudge');
const nudgeBtn        = document.getElementById('nudge-settings-btn');

// Mode Buttons
const modeAiBtn        = document.getElementById('mode-ai');
const modePrecisionBtn = document.getElementById('mode-precision');
const switchPrecBtn    = document.getElementById('switch-precision-btn');
const debugToggle      = document.getElementById('debug-toggle');

// Precision Mode Canvas Overlay
const precisionOverlay = document.getElementById('precision-overlay');
const precisionUndoBtn = document.getElementById('precision-undo');
const precisionCalcBtn = document.getElementById('precision-calculate');

// Initialize API Key UI
if (STATE.apiKey) {
  apiKeyInput.value = STATE.apiKey;
  noKeyNudge.classList.add('hidden');
} else {
  noKeyNudge.classList.remove('hidden');
}

// Open API Key Modal
function openApiModal() {
  apiKeyInput.value = STATE.apiKey;
  apiKeyStatus.textContent = '';
  apiModal.classList.remove('hidden');
  apiKeyInput.focus();
}

function closeApiModal() {
  apiModal.classList.add('hidden');
}

settingsBtn.addEventListener('click', openApiModal);
apiCloseBtn.addEventListener('click', closeApiModal);
apiKeyCancelBtn.addEventListener('click', closeApiModal);
nudgeBtn.addEventListener('click', openApiModal);

// Toggle Key Visibility
apiKeyToggle.addEventListener('click', () => {
  apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
});

// Save Key
apiKeySaveBtn.addEventListener('click', () => {
  const val = apiKeyInput.value.trim();
  if (!val) {
    apiKeyStatus.className = 'api-key-status error';
    apiKeyStatus.textContent = 'Please enter an API key.';
    return;
  }
  STATE.apiKey = val;
  localStorage.setItem('gemini_api_key', val);
  apiKeyStatus.className = 'api-key-status success';
  apiKeyStatus.textContent = 'Key saved securely in browser localStorage!';
  noKeyNudge.classList.add('hidden');
  setTimeout(closeApiModal, 800);
});

// Clear Key
apiKeyClearBtn.addEventListener('click', () => {
  STATE.apiKey = '';
  localStorage.removeItem('gemini_api_key');
  apiKeyInput.value = '';
  apiKeyStatus.className = 'api-key-status';
  apiKeyStatus.textContent = 'Key removed.';
  noKeyNudge.classList.remove('hidden');
});

// Mode Selection Handling
modeAiBtn.addEventListener('click', () => setMode('ai'));
modePrecisionBtn.addEventListener('click', () => setMode('precision'));
switchPrecBtn.addEventListener('click', () => setMode('precision'));

function setMode(newMode) {
  STATE.mode = newMode;
  if (newMode === 'ai') {
    modeAiBtn.classList.add('active');
    modeAiBtn.setAttribute('aria-pressed', 'true');
    modePrecisionBtn.classList.remove('active');
    modePrecisionBtn.setAttribute('aria-pressed', 'false');
    document.getElementById('precision-overlay').classList.add('hidden');
    document.getElementById('precision-controls').classList.add('hidden');

    if (STATE.rawImage && resultsSection.classList.contains('hidden') === false) {
      startAnalysis(STATE.rawImage);
    }
  } else {
    modePrecisionBtn.classList.add('active');
    modePrecisionBtn.setAttribute('aria-pressed', 'true');
    modeAiBtn.classList.remove('active');
    modeAiBtn.setAttribute('aria-pressed', 'false');

    if (STATE.rawImage) {
      initPrecisionMode(document.getElementById('result-canvas'));
    }
  }
}

// Debug Mode Toggle
debugToggle.addEventListener('change', (e) => {
  STATE.debugMode = e.target.checked;
  if (STATE.lastResult && STATE.rawImage) {
    renderOverlay(document.getElementById('result-canvas'), STATE.lastResult, STATE.debugMode);
  }
});

// Precision Overlay Click Handler
precisionOverlay.addEventListener('click', (e) => {
  if (STATE.precisionCorners.length >= 3) return;

  const canvas = document.getElementById('result-canvas');
  const rect = canvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Map to canvas coordinate system
  const canvasX = (clickX / rect.width) * canvas.width;
  const canvasY = (clickY / rect.height) * canvas.height;

  STATE.precisionCorners.push({ x: canvasX, y: canvasY });
  updatePrecisionUI();
});

// Precision Undo
precisionUndoBtn.addEventListener('click', () => {
  if (STATE.precisionCorners.length > 0) {
    STATE.precisionCorners.pop();
    updatePrecisionUI();
    const canvas = document.getElementById('result-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(STATE.rawImage, 0, 0, canvas.width, canvas.height);
  }
});

precisionCalcBtn.addEventListener('click', calculatePrecisionResult);

// File Upload & Drag-and-Drop
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) handleFile(f);
  fileInput.value = '';
});

cameraBtn.addEventListener('click', () => cameraInput.click());
cameraInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) handleFile(f);
  cameraInput.value = '';
});

// Sample Difficult Samosa Button
sampleBtn.addEventListener('click', async () => {
  try {
    showLoading();
    const res = await fetch('test_samosa.jpg');
    if (!res.ok) throw new Error('Could not fetch test samosa image.');
    const blob = await res.blob();
    const img = new Image();
    img.onload = () => startAnalysis(img);
    img.onerror = () => showError('Failed to load sample image.');
    img.src = URL.createObjectURL(blob);
  } catch (err) {
    showError('Sample photo error: ' + err.message);
  }
});

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
  else showError('Please drop a valid image file.');
});

retryBtn.addEventListener('click', resetUI);

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please upload an image file.');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError('Image too large (max 20 MB).');
    return;
  }
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => startAnalysis(img);
    img.onerror = () => showError('Could not load image.');
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function showLoading() {
  uploadSection.classList.add('hidden');
  loadingSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  resetSteps();
}

function showResults() {
  loadingSection.classList.add('hidden');
  resultsSection.classList.remove('hidden');
  resultsSection.classList.add('fade-in');
}

function resetUI() {
  uploadSection.classList.remove('hidden');
  loadingSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  resultsSection.classList.remove('fade-in');
  document.getElementById('precision-overlay').classList.add('hidden');
  document.getElementById('precision-controls').classList.add('hidden');
  STATE.precisionCorners = [];
  STATE.lastResult = null;
}

function resetSteps() {
  steps.forEach(s => s.classList.remove('active', 'done'));
}

function activateStep(idx) {
  steps.forEach((s, i) => {
    if (i < idx) {
      s.classList.remove('active');
      s.classList.add('done');
    } else if (i === idx) {
      s.classList.add('active');
      s.classList.remove('done');
    } else {
      s.classList.remove('active', 'done');
    }
  });
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.classList.remove('hidden');
  setTimeout(() => errorToast.classList.add('hidden'), 4500);
  loadingSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
