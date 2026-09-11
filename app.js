/**
 * Samosa Triangle Accuracy Finder — app.js
 * Complete Pipeline: Background Removal → Corner Detection → Triangle Accuracy
 *
 * 1. Background Removal:
 *    - Adaptive border background color modeling (distinguishes tables, plates, countertops)
 *    - Golden pastry crust HSV color space thresholding (Hue: 12°–58°, Saturation: 0.12–0.95, Value: 0.14–0.98)
 *    - Distance & difference metric against background palette
 *    - Central prior saliency weighting
 *    - Morphological Close (fuses crust blisters & folds) + Open (eliminates speckle noise)
 *    - Connected Component Analysis (selects primary central samosa blob)
 *    - Hole-filling via boundary flood fill
 *    - Generates transparent isolated samosa canvas (feathered alpha channel)
 *
 * 2. Corner Detection:
 *    - Contour extraction & Monotone Chain Convex Hull
 *    - Identifies 3 true physical corners:
 *      • Apex (Top tip) — Neon Red
 *      • Bottom-Left (BL) — Neon Green
 *      • Bottom-Right (BR) — Neon Blue
 *    - Local exterior curvature & sharpness angle optimization
 *
 * 3. Mathematical Triangle Accuracy (Infographic Formula):
 *    - Side lengths: a = BC, b = CA, c = AB
 *    - Average side length: s = (a + b + c) / 3
 *    - Side delta: diff = |a - b| + |b - c| + |c - a|
 *    - Accuracy = (1 - diff / (3 * s)) * 100
 *    - Interior angles: Law of Cosines (A, B, C; ideal = 60° each)
 *
 * 4. Interactive Views & 4-Step Stepper:
 *    - View Switcher: Isolated Samosa (No BG) | Original Photo | Samosa Mask
 *    - 4 Pipeline Preview Steps: 1. Original → 2. BG Removal → 3. Corners → 4. Accuracy
 */

'use strict';

/* ═══════════════════════════════════════════════════════
   0. CONFIGURATION & STATE
   ═══════════════════════════════════════════════════════ */
const CONFIG = {
  maxProcessDim: 720,        // standard dimension for fast CV & AI inference
  geminiModel: 'gemini-2.5-flash',
  fallbackModel: 'gemini-2.0-flash',
  confidenceThreshold: 0.55,
  localSearchRadiusPct: 0.15 // fraction of samosa size for corner snapping
};

const STATE = {
  apiKey: localStorage.getItem('gemini_api_key') || '',
  mode: 'ai',                // 'ai' | 'precision'
  debugMode: false,
  viewMode: 'nobg',          // 'nobg' | 'original' | 'mask'
  pipelineStage: 4,          // 1: Original, 2: BG Removed, 3: Corners, 4: Accuracy
  rawImage: null,            // Source Image
  scaledCanvas: null,        // Resized processing canvas
  scaleRatio: 1.0,
  lastResult: null,          // Caches full analysis results
  precisionCorners: [],      // [{x, y}] manual points
  analyzing: false
};

/* ═══════════════════════════════════════════════════════
   1. COLOR UTILITIES
   ═══════════════════════════════════════════════════════ */

/** Converts RGB [0..255] to HSV: H in [0..360), S in [0..1], V in [0..1] */
function rgbToHsv(r, g, b) {
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;
  let h = 0;
  const s = max === 0 ? 0 : d / max;
  const v = max;

  if (d !== 0) {
    if (max === rf) {
      h = 60 * (((gf - bf) / d) % 6);
    } else if (max === gf) {
      h = 60 * (((bf - rf) / d) + 2);
    } else {
      h = 60 * (((rf - gf) / d) + 4);
    }
    if (h < 0) h += 360;
  }
  return { h, s, v };
}

/* ═══════════════════════════════════════════════════════
   2. MATHEMATICAL GEOMETRY & ACCURACY ENGINE
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
  const a = dist(B, C); // opposite to A (base side)
  const b = dist(A, C); // opposite to B (right side)
  const c = dist(A, B); // opposite to C (left side)

  if (a <= 0 || b <= 0 || c <= 0) {
    return { A: 60, B: 60, C: 60, sum: 180, sideA: a, sideB: b, sideC: c, valid: false };
  }

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
    sideA: a,
    sideB: b,
    sideC: c,
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
 * Computes full geometry & accuracy score using the infographic formula:
 * Accuracy = (1 - (|a - b| + |b - c| + |c - a|) / (3 * s)) * 100
 */
function computeGeometryResults(triangle, samosaMask, hullPts, W, H) {
  const [A, B, C] = triangle;
  const geom = computeAnglesFromVertices(A, B, C);
  const triAreaVal = triangleArea(A, B, C);

  // Measure samosa silhouette area
  let contourArea = 0;
  if (samosaMask) {
    for (let i = 0; i < samosaMask.length; i++) {
      if (samosaMask[i] > 0) contourArea++;
    }
  } else if (hullPts && hullPts.length >= 3) {
    contourArea = polygonArea(hullPts);
  } else {
    contourArea = triAreaVal;
  }

  // Rasterize triangle to calculate IoU against samosa mask
  let intersection = 0;
  let union = 0;
  if (samosaMask && W > 0 && H > 0) {
    const triMask = rasterizeTriangle(A, B, C, W, H);
    for (let i = 0; i < W * H; i++) {
      const isTri = triMask[i] > 0;
      const isSam = samosaMask[i] > 0;
      if (isTri && isSam) intersection++;
      if (isTri || isSam) union++;
    }
  }

  const iou = union > 0 ? (intersection / union) * 100 : 70;
  const areaRatio = contourArea > 0 ? Math.min(triAreaVal / contourArea, contourArea / triAreaVal) * 100 : 70;

  // Infographic Mathematical Formula:
  // a = BC, b = CA, c = AB
  const a = geom.sideA;
  const b = geom.sideB;
  const c = geom.sideC;
  const s = (a + b + c) / 3;
  const sideDiff = Math.abs(a - b) + Math.abs(b - c) + Math.abs(c - a);
  let formulaAccuracy = s > 0 ? (1 - sideDiff / (3 * s)) * 100 : 0;
  formulaAccuracy = Math.max(0, Math.min(100, Math.round(formulaAccuracy * 10) / 10));

  // Equilateral Angle Deviation (ideal: 60° each)
  const angleDev = (Math.abs(geom.A - 60) + Math.abs(geom.B - 60) + Math.abs(geom.C - 60)) / 3;
  const angleSymmetryScore = Math.max(0, 100 - angleDev * 1.5);

  // Blended Accuracy Score (Primary: Infographic Formula 80% + Angle Symmetry 20%)
  let accuracyScore = Math.round(formulaAccuracy * 0.80 + angleSymmetryScore * 0.20);
  accuracyScore = Math.max(8, Math.min(99, accuracyScore));

  return {
    vertices: { apex: A, bottomLeft: B, bottomRight: C },
    angles: { A: geom.A, B: geom.B, C: geom.C, sum: geom.sum },
    sideLengths: { AB: c, BC: a, CA: b },
    formula: {
      a: a,
      b: b,
      c: c,
      s: s,
      sideDiff: sideDiff,
      formulaAccuracy: formulaAccuracy,
      angleDev: angleDev
    },
    triArea: triAreaVal,
    contourArea: contourArea,
    areaRatio: areaRatio,
    iou: iou,
    angleSymmetry: angleSymmetryScore,
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
   3. BACKGROUND REMOVAL & SEGMENTATION ENGINE
   ═══════════════════════════════════════════════════════ */

/**
 * Removes background from the image canvas and isolates the samosa.
 *
 * Techniques:
 *  1. Border Background Color Profiling (identifies tables, plates, countertops)
 *  2. HSV Golden Crust Segmenter (Hue 12°–58°, Saturation 0.12–0.95, Value 0.14–0.98)
 *  3. Difference & Distance metric against background border palette
 *  4. Central Prior Saliency Weighting
 *  5. Morphological Close + Open
 *  6. Connected Component Analysis (selects the primary central samosa blob)
 *  7. Hole-filling via boundary flood fill
 *  8. Transparent Isolated Canvas Generation (with feathered alpha)
 */
function removeBackground(sourceCanvas, bbox = null) {
  const W = sourceCanvas.width, H = sourceCanvas.height;
  const ctx = sourceCanvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, W, H);
  const data = imgData.data;

  // 1. Check if image already has a transparent background (e.g., PNG with alpha)
  let transparentPixelCount = 0;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 50) transparentPixelCount++;
  }
  const hasExistingTransparency = transparentPixelCount > (W * H * 0.04);

  // 2. Profile border background colors (outer 6% perimeter)
  const borderMarginX = Math.max(3, Math.floor(W * 0.06));
  const borderMarginY = Math.max(3, Math.floor(H * 0.06));
  let bgRSum = 0, bgGSum = 0, bgBSum = 0, borderCount = 0;
  const borderColors = [];

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (x < borderMarginX || x >= W - borderMarginX || y < borderMarginY || y >= H - borderMarginY) {
        const idx = (y * W + x) * 4;
        if (data[idx + 3] > 100) {
          const r = data[idx], g = data[idx + 1], b = data[idx + 2];
          bgRSum += r;
          bgGSum += g;
          bgBSum += b;
          borderCount++;
          if (borderCount % 8 === 0) borderColors.push({ r, g, b });
        }
      }
    }
  }

  const avgBgR = borderCount > 0 ? bgRSum / borderCount : 240;
  const avgBgG = borderCount > 0 ? bgGSum / borderCount : 240;
  const avgBgB = borderCount > 0 ? bgBSum / borderCount : 240;

  // 3. Define bounding box bounds if provided (e.g. from AI)
  const xMin = bbox ? Math.max(0, bbox.xmin) : 0;
  const xMax = bbox ? Math.min(W - 1, bbox.xmax) : W - 1;
  const yMin = bbox ? Math.max(0, bbox.ymin) : 0;
  const yMax = bbox ? Math.min(H - 1, bbox.ymax) : H - 1;

  const rawMask = new Uint8Array(W * H);
  const centerX = W / 2, centerY = H / 2;
  const maxCenterDist = Math.hypot(centerX, centerY);

  // 4. Pixel classification (HSV + Saliency + Distance from background)
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;

      // Honor existing transparency
      if (hasExistingTransparency) {
        if (data[idx + 3] > 80) rawMask[y * W + x] = 255;
        continue;
      }

      // Restrict to bbox if given
      if (x < xMin || x > xMax || y < yMin || y > yMax) {
        continue;
      }

      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const hsv = rgbToHsv(r, g, b);

      // Distance from average background
      const distFromBg = Math.hypot(r - avgBgR, g - avgBgG, b - avgBgB);

      // Distance from closest sample in border palette
      let minDistToBorderSample = Infinity;
      const stepCheck = Math.max(1, Math.floor(borderColors.length / 30));
      for (let k = 0; k < borderColors.length; k += stepCheck) {
        const bc = borderColors[k];
        const d = Math.hypot(r - bc.r, g - bc.g, b - bc.b);
        if (d < minDistToBorderSample) minDistToBorderSample = d;
      }

      // Saliency center prior
      const centerDistNorm = Math.hypot(x - centerX, y - centerY) / maxCenterDist;

      // Fried pastry color characteristics:
      // Golden yellow/orange/amber hue
      const isPastryHue = (hsv.h >= 10 && hsv.h <= 60);
      const isPastrySat = (hsv.s >= 0.12 && hsv.s <= 0.96);
      const isPastryVal = (hsv.v >= 0.14 && hsv.v <= 0.98);
      const isWarm = (r > b + 10) && (r >= g * 0.82) && (r > 50);

      // Background rejection tests:
      const isDifferentFromBg = (distFromBg > 22 && minDistToBorderSample > 18);
      const isNeutralPlate = (hsv.s < 0.10) && (r > 190 && g > 190 && b > 190);
      const isBlackShadow = (hsv.v < 0.10);

      if (isWarm && isPastryHue && isPastrySat && isPastryVal && isDifferentFromBg && !isNeutralPlate && !isBlackShadow) {
        // Border decay: require higher warmth near edge
        if (centerDistNorm > 0.85 && (!bbox)) {
          if (isWarm && hsv.s > 0.25 && distFromBg > 35) {
            rawMask[y * W + x] = 255;
          }
        } else {
          rawMask[y * W + x] = 255;
        }
      }
    }
  }

  // 5. Morphological operations: Close (radius 4) to bridge ajwain seeds and cracks, Open (radius 2) to drop speckles
  const closedMask = morphClose(rawMask, W, H, 4);
  const cleanedMask = morphOpen(closedMask, W, H, 2);

  // 6. Connected Component Analysis — Keep primary central samosa blob
  const { filteredMask, primaryBlob } = extractPrimaryComponent(cleanedMask, W, H);

  // 7. Hole filling: flood fill background from outer perimeter, invert to make samosa solid
  const solidMask = fillMaskHoles(filteredMask, W, H);

  // 8. Generate transparent isolated samosa canvas
  const isolatedCanvas = document.createElement('canvas');
  isolatedCanvas.width = W;
  isolatedCanvas.height = H;
  const ictx = isolatedCanvas.getContext('2d');
  const isolatedImgData = ictx.createImageData(W, H);
  const idata = isolatedImgData.data;

  // Soft edge / feathered alpha
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      const maskVal = solidMask[y * W + x];
      if (maskVal > 0) {
        idata[idx] = data[idx];
        idata[idx + 1] = data[idx + 1];
        idata[idx + 2] = data[idx + 2];
        idata[idx + 3] = 255;
      } else {
        idata[idx] = 0;
        idata[idx + 1] = 0;
        idata[idx + 2] = 0;
        idata[idx + 3] = 0;
      }
    }
  }
  ictx.putImageData(isolatedImgData, 0, 0);

  // 9. Generate Mask Canvas (Neon Cyan on dark)
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = W;
  maskCanvas.height = H;
  const mctx = maskCanvas.getContext('2d');
  const maskImgData = mctx.createImageData(W, H);
  const mdata = maskImgData.data;

  for (let i = 0; i < W * H; i++) {
    const idx = i * 4;
    if (solidMask[i] > 0) {
      mdata[idx] = 0;
      mdata[idx + 1] = 212;
      mdata[idx + 2] = 255;
      mdata[idx + 3] = 220;
    } else {
      mdata[idx] = 12;
      mdata[idx + 1] = 12;
      mdata[idx + 2] = 20;
      mdata[idx + 3] = 255;
    }
  }
  mctx.putImageData(maskImgData, 0, 0);

  // 10. Extract outer boundary points and Convex Hull
  const boundaryPts = extractBoundaryPoints(solidMask, W, H);
  const hullPts = boundaryPts.length >= 3 ? convexHull(boundaryPts) : null;

  return {
    isolatedCanvas: isolatedCanvas,
    maskCanvas: maskCanvas,
    mask: solidMask,
    boundaryPts: boundaryPts,
    hullPts: hullPts,
    primaryBlob: primaryBlob,
    hasCleanSegmentation: (boundaryPts.length >= 10)
  };
}

/** 8-Connected Component Analysis to pick the prominent central samosa */
function extractPrimaryComponent(mask, W, H) {
  const labels = new Int32Array(W * H);
  let currentLabel = 1;
  const blobStats = [];

  const queue = [];

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      if (mask[idx] === 255 && labels[idx] === 0) {
        let area = 0;
        let sumX = 0, sumY = 0;
        let touchesBorder = false;

        labels[idx] = currentLabel;
        queue.push(x, y);

        let head = 0;
        while (head < queue.length) {
          const qx = queue[head++];
          const qy = queue[head++];
          area++;
          sumX += qx;
          sumY += qy;

          if (qx <= 2 || qx >= W - 3 || qy <= 2 || qy >= H - 3) {
            touchesBorder = true;
          }

          // 4-neighbors
          const neighbors = [
            [qx + 1, qy],
            [qx - 1, qy],
            [qx, qy + 1],
            [qx, qy - 1]
          ];

          for (const [nx, ny] of neighbors) {
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
              const nIdx = ny * W + nx;
              if (mask[nIdx] === 255 && labels[nIdx] === 0) {
                labels[nIdx] = currentLabel;
                queue.push(nx, ny);
              }
            }
          }
        }
        queue.length = 0;

        const centroidX = sumX / area;
        const centroidY = sumY / area;
        const distFromCenter = Math.hypot(centroidX - W / 2, centroidY - H / 2);
        const centerScore = Math.max(0.1, 1.0 - (distFromCenter / Math.hypot(W / 2, H / 2)));

        blobStats.push({
          label: currentLabel,
          area: area,
          centroidX: centroidX,
          centroidY: centroidY,
          touchesBorder: touchesBorder,
          score: area * (centerScore ** 1.3)
        });

        currentLabel++;
      }
    }
  }

  if (blobStats.length === 0) {
    return { filteredMask: mask, primaryBlob: null };
  }

  // Sort by score (area + central location)
  blobStats.sort((a, b) => b.score - a.score);
  const bestBlob = blobStats[0];

  // Build binary mask of only the primary blob
  const filteredMask = new Uint8Array(W * H);
  const targetLabel = bestBlob.label;

  for (let i = 0; i < W * H; i++) {
    if (labels[i] === targetLabel) {
      filteredMask[i] = 255;
    }
  }

  return { filteredMask: filteredMask, primaryBlob: bestBlob };
}

/** Fills any internal holes in the mask using flood fill from image perimeter */
function fillMaskHoles(mask, W, H) {
  const inverted = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    inverted[i] = mask[i] === 0 ? 1 : 0;
  }

  const reached = new Uint8Array(W * H);
  const queue = [];

  // Seed boundary pixels
  for (let x = 0; x < W; x++) {
    if (inverted[x] === 1 && !reached[x]) { reached[x] = 1; queue.push(x, 0); }
    const bIdx = (H - 1) * W + x;
    if (inverted[bIdx] === 1 && !reached[bIdx]) { reached[bIdx] = 1; queue.push(x, H - 1); }
  }
  for (let y = 0; y < H; y++) {
    const lIdx = y * W;
    if (inverted[lIdx] === 1 && !reached[lIdx]) { reached[lIdx] = 1; queue.push(0, y); }
    const rIdx = y * W + (W - 1);
    if (inverted[rIdx] === 1 && !reached[rIdx]) { reached[rIdx] = 1; queue.push(W - 1, y); }
  }

  let head = 0;
  while (head < queue.length) {
    const qx = queue[head++];
    const qy = queue[head++];

    const neighbors = [
      [qx + 1, qy],
      [qx - 1, qy],
      [qx, qy + 1],
      [qx, qy - 1]
    ];

    for (const [nx, ny] of neighbors) {
      if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
        const nIdx = ny * W + nx;
        if (inverted[nIdx] === 1 && reached[nIdx] === 0) {
          reached[nIdx] = 1;
          queue.push(nx, ny);
        }
      }
    }
  }

  // Any pixel not reached from the borders is an interior hole! Fill it!
  const filled = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (reached[i] === 0) {
      filled[i] = 255;
    }
  }

  return filled;
}

/** Extracts boundary contour points of solid mask */
function extractBoundaryPoints(mask, W, H) {
  const pts = [];
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const idx = y * W + x;
      if (mask[idx] === 255) {
        if (
          mask[idx - 1] === 0 ||
          mask[idx + 1] === 0 ||
          mask[idx - W] === 0 ||
          mask[idx + W] === 0
        ) {
          pts.push({ x, y });
        }
      }
    }
  }
  return pts;
}

/** Morphological Close (Dilation followed by Erosion) */
function morphClose(src, w, h, radius = 3) {
  return morphErode(morphDilate(src, w, h, radius), w, h, radius);
}

/** Morphological Open (Erosion followed by Dilation) */
function morphOpen(src, w, h, radius = 2) {
  return morphDilate(morphErode(src, w, h, radius), w, h, radius);
}

function morphDilate(src, w, h, radius = 2) {
  const dst = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let maxVal = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          if (src[ny * w + nx] > maxVal) { maxVal = 255; break; }
        }
        if (maxVal === 255) break;
      }
      dst[y * w + x] = maxVal;
    }
  }
  return dst;
}

function morphErode(src, w, h, radius = 2) {
  const dst = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let minVal = 255;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          if (src[ny * w + nx] === 0) { minVal = 0; break; }
        }
        if (minVal === 0) break;
      }
      dst[y * w + x] = minVal;
    }
  }
  return dst;
}

/** Convex Hull via Monotone Chain Algorithm */
function convexHull(pts) {
  if (pts.length <= 3) return pts.slice();
  const sorted = pts.slice().sort((a, b) => (a.x !== b.x ? a.x - b.x : a.y - b.y));

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
   4. CORNER DETECTION ENGINE (ON ISOLATED SAMOSA)
   ═══════════════════════════════════════════════════════ */

/**
 * Detects the 3 outer corners (Apex, Bottom-Left, Bottom-Right) on the clean samosa hull.
 * Matches infographic:
 *   • Apex: Top tip (Red circle)
 *   • Bottom-Left: Left base corner (Green circle)
 *   • Bottom-Right: Right base corner (Blue circle)
 */
function detectCornersFromSamosa(hullPts, W, H) {
  if (!hullPts || hullPts.length < 3) {
    return [
      { x: W * 0.5, y: H * 0.18 },
      { x: W * 0.2, y: H * 0.82 },
      { x: W * 0.8, y: H * 0.82 }
    ];
  }

  const n = hullPts.length;

  // 1. Find the centroid of the hull
  let cx = 0, cy = 0;
  for (const p of hullPts) { cx += p.x; cy += p.y; }
  cx /= n; cy /= n;

  // 2. Identify candidate corners:
  // Apex: minimum Y (uppermost tip)
  let bestApex = hullPts[0];
  let minApexY = Infinity;

  // Bottom-Left: corner maximizing (-x + y)
  let bestBL = hullPts[0];
  let maxBLScore = -Infinity;

  // Bottom-Right: corner maximizing (x + y)
  let bestBR = hullPts[0];
  let maxBRScore = -Infinity;

  for (let i = 0; i < n; i++) {
    const p = hullPts[i];

    // Local exterior turn sharpness along the hull
    const prev = hullPts[(i - 2 + n) % n];
    const next = hullPts[(i + 2) % n];
    const v1 = { x: prev.x - p.x, y: prev.y - p.y };
    const v2 = { x: next.x - p.x, y: next.y - p.y };
    const len1 = Math.hypot(v1.x, v1.y) || 1;
    const len2 = Math.hypot(v2.x, v2.y) || 1;
    const dot = (v1.x * v2.x + v1.y * v2.y) / (len1 * len2);
    const sharpness = 1.0 - dot; // range 0..2 (higher = sharper turn)

    // Apex score (favors top Y, upward protrusion away from centroid, and sharpness)
    const apexScore = -p.y * 1.5 + (cy - p.y) * 0.5 + sharpness * 25;
    if (apexScore > -minApexY) {
      minApexY = -apexScore;
      bestApex = p;
    }

    // Bottom-Left score (left of centroid, low in image, sharp)
    const blScore = -(p.x - cx) * 1.3 + (p.y - cy) * 1.1 + sharpness * 20;
    if (blScore > maxBLScore) {
      maxBLScore = blScore;
      bestBL = p;
    }

    // Bottom-Right score (right of centroid, low in image, sharp)
    const brScore = (p.x - cx) * 1.3 + (p.y - cy) * 1.1 + sharpness * 20;
    if (brScore > maxBRScore) {
      maxBRScore = brScore;
      bestBR = p;
    }
  }

  // Refine points to snap locally onto the sharpest local vertex along the hull
  const refineRadius = Math.max(12, Math.hypot(W, H) * 0.08);
  const refinedApex = findSharpestHullPoint(bestApex, hullPts, refineRadius, 'top');
  const refinedBL   = findSharpestHullPoint(bestBL, hullPts, refineRadius, 'bottomLeft');
  const refinedBR   = findSharpestHullPoint(bestBR, hullPts, refineRadius, 'bottomRight');

  return [refinedApex, refinedBL, refinedBR];
}

function findSharpestHullPoint(targetPt, hullPts, maxRadius, role) {
  let bestPt = targetPt;
  let bestScore = -Infinity;
  const n = hullPts.length;

  for (let i = 0; i < n; i++) {
    const pt = hullPts[i];
    const d = dist(pt, targetPt);
    if (d > maxRadius) continue;

    const prev = hullPts[(i - 2 + n) % n];
    const next = hullPts[(i + 2) % n];
    const v1 = { x: prev.x - pt.x, y: prev.y - pt.y };
    const v2 = { x: next.x - pt.x, y: next.y - pt.y };
    const len1 = Math.hypot(v1.x, v1.y) || 1;
    const len2 = Math.hypot(v2.x, v2.y) || 1;
    const dot = (v1.x * v2.x + v1.y * v2.y) / (len1 * len2);
    const curvature = 1.0 - dot;

    let dirBonus = 0;
    if (role === 'top') dirBonus = -pt.y * 0.05;
    else if (role === 'bottomLeft') dirBonus = -pt.x * 0.04 + pt.y * 0.03;
    else if (role === 'bottomRight') dirBonus = pt.x * 0.04 + pt.y * 0.03;

    const distPenalty = (d / maxRadius) * 0.6;
    const score = curvature + dirBonus - distPenalty;

    if (score > bestScore) {
      bestScore = score;
      bestPt = pt;
    }
  }
  return bestPt;
}

/* ═══════════════════════════════════════════════════════
   5. GEMINI VISION AI DETECTOR
   ═══════════════════════════════════════════════════════ */

async function callGeminiVision(canvas) {
  if (!STATE.apiKey) throw new Error('NO_API_KEY');

  const base64Data = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
  const prompt = `You are a computer vision model for samosa detection.
Your task is to detect the main samosa in the image and locate its bounding box and 3 physical corners.
DO NOT guess or calculate angles or accuracy scores. ONLY return coordinates.

Exclusion Rules:
- Exclude the background, plates, table, and hands holding the samosa.
- The corners must be the 3 outer corners of the actual samosa pastry:
  "apex": The top tip/peak of the samosa.
  "bottomLeft": The bottom-left corner.
  "bottomRight": The bottom-right corner.

Return strictly JSON:
{
  "hasSamosa": true,
  "confidence": 0.94,
  "boundingBox": { "ymin": 100, "xmin": 150, "ymax": 900, "xmax": 850 },
  "corners": {
    "apex": { "x": 500, "y": 120 },
    "bottomLeft": { "x": 200, "y": 850 },
    "bottomRight": { "x": 800, "y": 830 }
  }
}`;

  const requestBody = {
    contents: [
      {
        parts: [
          { text: prompt },
          { inline_data: { mime_type: 'image/jpeg', data: base64Data } }
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
        throw new Error(errData.error?.message || `HTTP ${response.status}`);
      }

      const resJson = await response.json();
      const rawText = resJson.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!rawText) throw new Error('No content returned from Gemini Vision API.');

      const cleanJson = rawText.replace(/```json/gi, '').replace(/```/g, '').trim();
      const result = JSON.parse(cleanJson);
      if (!result.hasSamosa || !result.corners) {
        throw new Error('No samosa recognized by the AI in this image.');
      }
      return result;
    } catch (err) {
      console.warn(`[AI] Model ${model} failed:`, err.message);
      lastError = err;
    }
  }
  throw lastError || new Error('Failed to reach Gemini Vision API.');
}

/* ═══════════════════════════════════════════════════════
   6. OVERLAY RENDERING & 4-STEP PIPELINE PREVIEW
   ═══════════════════════════════════════════════════════ */

/**
 * Renders the analyzed image based on the selected View Mode and Pipeline Stage:
 *
 * View Modes:
 *  • 'nobg' (default): Samosa isolated on checkerboard background (matches infographic)
 *  • 'original': Original photograph with overlay
 *  • 'mask': Segmentation mask view
 *
 * Pipeline Stages:
 *  • Stage 1: Original Image
 *  • Stage 2: Background Removal (Clean isolated samosa on checkerboard)
 *  • Stage 3: Corner Detection (Apex in Red, BL in Green, BR in Blue, Contour)
 *  • Stage 4: Measurement & Accuracy (Triangle + angles + side lengths + score)
 */
function renderOverlay(canvas, result, viewMode = 'nobg', stage = 4, showDebug = false) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  // 1. Draw base layer
  if (stage === 1 || viewMode === 'original') {
    // Draw original raw photo
    ctx.drawImage(STATE.rawImage, 0, 0, W, H);
  } else if (viewMode === 'mask') {
    // Draw segmentation mask
    if (result.maskCanvas) {
      ctx.drawImage(result.maskCanvas, 0, 0, W, H);
    } else {
      ctx.drawImage(STATE.rawImage, 0, 0, W, H);
    }
  } else {
    // 'nobg' mode: Draw subtle checkerboard + isolated transparent samosa
    drawCheckerboard(ctx, W, H);
    if (result.isolatedCanvas) {
      ctx.drawImage(result.isolatedCanvas, 0, 0, W, H);
    } else {
      ctx.drawImage(STATE.rawImage, 0, 0, W, H);
    }
  }

  // If stage 1 or stage 2, stop here (clean preview of step 1 or step 2)
  if (stage === 1) return;
  if (stage === 2) return;

  const [A, B, C] = result.refinedCorners;
  const geom = result.geometry;
  const angles = geom.angles;

  // 2. Debug Layers (optional)
  if (showDebug) {
    if (result.bbox) {
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.75)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(result.bbox.xmin, result.bbox.ymin, result.bbox.xmax - result.bbox.xmin, result.bbox.ymax - result.bbox.ymin);
      ctx.fillStyle = '#00d4ff';
      ctx.font = 'bold 11px Outfit, sans-serif';
      ctx.fillText('SAMOSA BOUNDS', result.bbox.xmin + 6, result.bbox.ymin + 14);
      ctx.restore();
    }
    if (result.rawAiCorners) {
      result.rawAiCorners.forEach((pt, i) => {
        ctx.save();
        ctx.strokeStyle = '#ff00ea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      });
    }
  }

  // 3. Draw Samosa Outer Contour
  if (result.hullPts && result.hullPts.length > 2) {
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.65)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(result.hullPts[0].x, result.hullPts[0].y);
    for (let i = 1; i < result.hullPts.length; i++) {
      ctx.lineTo(result.hullPts[i].x, result.hullPts[i].y);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  // 4. In Stage 3 (Corner Detection): Draw corners & connecting edges
  if (stage >= 3) {
    // Triangle fill & edges
    ctx.save();
    ctx.fillStyle = 'rgba(57, 217, 138, 0.14)';
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.lineTo(C.x, C.y);
    ctx.closePath();
    ctx.fill();

    // Vibrant Green triangle boundary (matches infographic)
    ctx.strokeStyle = '#39d98a';
    ctx.lineWidth = 3.5;
    ctx.shadowColor = '#39d98a';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.lineTo(C.x, C.y);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();

    // Side lengths badges along edges (Stage 4)
    if (stage === 4) {
      drawSideBadge(ctx, A, B, `c: ${Math.round(geom.sideLengths.AB)}px`);
      drawSideBadge(ctx, B, C, `a: ${Math.round(geom.sideLengths.BC)}px`);
      drawSideBadge(ctx, C, A, `b: ${Math.round(geom.sideLengths.CA)}px`);

      // Draw angle arcs at each corner
      drawAngleArc(ctx, A, B, C, '#ff3b30');
      drawAngleArc(ctx, B, A, C, '#34c759');
      drawAngleArc(ctx, C, A, B, '#007aff');
    }

    // Vertices: Exact infographic colors
    // Apex = Red, Bottom-Left = Green, Bottom-Right = Blue
    const vertices = [
      { pt: A, name: 'A', role: 'Apex', angle: angles.A, color: '#ff3b30' },
      { pt: B, name: 'B', role: 'BL',   angle: angles.B, color: '#34c759' },
      { pt: C, name: 'C', role: 'BR',   angle: angles.C, color: '#007aff' }
    ];

    vertices.forEach(v => {
      // Glow circle
      ctx.save();
      ctx.fillStyle = v.color;
      ctx.shadowColor = v.color;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(v.pt.x, v.pt.y, 7.5, 0, Math.PI * 2);
      ctx.fill();

      // Inner white center
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(v.pt.x, v.pt.y, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Angle label pill (Stage 4)
      if (stage === 4) {
        drawAnglePill(ctx, v.pt.x, v.pt.y, `${v.name}: ~${v.angle.toFixed(1)}°`, v.color);
      }
    });
  }
}

/** Draws subtle transparency checkerboard pattern */
function drawCheckerboard(ctx, w, h, size = 16) {
  ctx.save();
  ctx.fillStyle = '#0c0c16';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#151522';
  for (let y = 0; y < h; y += size) {
    for (let x = 0; x < w; x += size) {
      if (((x / size) + (y / size)) % 2 === 0) {
        ctx.fillRect(x, y, size, size);
      }
    }
  }
  ctx.restore();
}

/** Draws angle arc indicator between two rays at vertex V */
function drawAngleArc(ctx, V, P1, P2, color = '#39d98a', radius = 22) {
  const ang1 = Math.atan2(P1.y - V.y, P1.x - V.x);
  const ang2 = Math.atan2(P2.y - V.y, P2.x - V.x);

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  let diff = ang2 - ang1;
  while (diff < -Math.PI) diff += 2 * Math.PI;
  while (diff > Math.PI) diff -= 2 * Math.PI;

  if (diff > 0) {
    ctx.arc(V.x, V.y, radius, ang1, ang1 + diff);
  } else {
    ctx.arc(V.x, V.y, radius, ang2, ang2 - diff);
  }
  ctx.stroke();
  ctx.restore();
}

/** Draws side length badge at midpoint of segment */
function drawSideBadge(ctx, p1, p2, text) {
  const mx = (p1.x + p2.x) / 2;
  const my = (p1.y + p2.y) / 2;

  ctx.save();
  ctx.font = '600 11px Outfit, sans-serif';
  const pad = 5;
  const tw = ctx.measureText(text).width;
  const pw = tw + pad * 2;
  const ph = 18;

  ctx.fillStyle = 'rgba(10, 10, 20, 0.85)';
  ctx.strokeStyle = 'rgba(57, 217, 138, 0.5)';
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.roundRect(mx - pw / 2, my - ph / 2, pw, ph, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#39d98a';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, mx, my);
  ctx.restore();
}

/** Draws rounded glass pill for angle measurement on canvas */
function drawAnglePill(ctx, x, y, text, accentColor = '#ffd700') {
  ctx.save();
  ctx.font = 'bold 12px Outfit, sans-serif';
  const pad = 6;
  const tw = ctx.measureText(text).width;
  const pw = tw + pad * 2;
  const ph = 22;
  const px = x + 10;
  const py = y - 10;

  ctx.fillStyle = 'rgba(10, 10, 20, 0.90)';
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1.4;

  ctx.beginPath();
  ctx.roundRect(px, py - ph + 4, pw, ph, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = accentColor;
  ctx.fillText(text, px + pad, py);
  ctx.restore();
}

/* ═══════════════════════════════════════════════════════
   7. UI RENDERING & MEASUREMENTS DISPLAY
   ═══════════════════════════════════════════════════════ */

function renderMeasurements(geom) {
  const f = geom.formula;
  const grid = document.getElementById('measurement-grid');

  grid.innerHTML = `
    <div class="m-item">
      <div class="m-label">Side AB (Left)</div>
      <div class="m-value highlight">${Math.round(geom.sideLengths.AB)} px</div>
    </div>
    <div class="m-item">
      <div class="m-label">Side BC (Base)</div>
      <div class="m-value highlight">${Math.round(geom.sideLengths.BC)} px</div>
    </div>
    <div class="m-item">
      <div class="m-label">Side CA (Right)</div>
      <div class="m-value highlight">${Math.round(geom.sideLengths.CA)} px</div>
    </div>
    <div class="m-item">
      <div class="m-label">Average Side (s)</div>
      <div class="m-value">${Math.round(f.s)} px</div>
    </div>
    <div class="m-item">
      <div class="m-label">Interior Angles</div>
      <div class="m-value">${geom.angles.A.toFixed(1)}° · ${geom.angles.B.toFixed(1)}° · ${geom.angles.C.toFixed(1)}°</div>
    </div>
    <div class="m-item">
      <div class="m-label">Angle Sum (180° Check)</div>
      <div class="m-value">${geom.angles.sum.toFixed(1)}° ${Math.abs(geom.angles.sum - 180) < 0.5 ? '✓' : ''}</div>
    </div>
    <div class="m-item">
      <div class="m-label">Equilateral Balance</div>
      <div class="m-value">${f.formulaAccuracy.toFixed(1)}%</div>
    </div>
    <div class="m-item">
      <div class="m-label">Shape Match (IoU)</div>
      <div class="m-value">${Math.round(geom.iou)}%</div>
    </div>
  `;

  // Formula Breakdown Details (live calculation)
  const formulaDetails = document.getElementById('formula-details');
  if (formulaDetails && f) {
    formulaDetails.innerHTML = `
      <div><strong>Sides:</strong> a = ${f.a.toFixed(1)}px · b = ${f.b.toFixed(1)}px · c = ${f.c.toFixed(1)}px</div>
      <div><strong>Average (s):</strong> ${f.s.toFixed(1)}px &nbsp;|&nbsp; <strong>Side Delta:</strong> ${f.sideDiff.toFixed(1)}px</div>
      <div><strong>Accuracy:</strong> (1 − ${f.sideDiff.toFixed(1)} / ${(3 * f.s).toFixed(1)}) × 100 = <span style="color:#ffd700;font-weight:bold">${f.formulaAccuracy.toFixed(1)}%</span></div>
    `;
  }

  // Vertices Table with colored dots
  const tbody = document.getElementById('vertices-body');
  tbody.innerHTML = `
    <tr>
      <td><span class="dot dot-red" style="vertical-align:middle;margin-right:5px"></span><strong>A (Apex)</strong></td>
      <td>${Math.round(geom.vertices.apex.x)}</td>
      <td>${Math.round(geom.vertices.apex.y)}</td>
      <td>${geom.angles.A.toFixed(1)}°</td>
    </tr>
    <tr>
      <td><span class="dot dot-green" style="vertical-align:middle;margin-right:5px"></span><strong>B (Bottom-Left)</strong></td>
      <td>${Math.round(geom.vertices.bottomLeft.x)}</td>
      <td>${Math.round(geom.vertices.bottomLeft.y)}</td>
      <td>${geom.angles.B.toFixed(1)}°</td>
    </tr>
    <tr>
      <td><span class="dot dot-blue" style="vertical-align:middle;margin-right:5px"></span><strong>C (Bottom-Right)</strong></td>
      <td>${Math.round(geom.vertices.bottomRight.x)}</td>
      <td>${Math.round(geom.vertices.bottomRight.y)}</td>
      <td>${geom.angles.C.toFixed(1)}°</td>
    </tr>
  `;
}

function renderScore(score) {
  const numEl = document.getElementById('score-number');
  const labelEl = document.getElementById('result-label');
  const ringFill = document.getElementById('ring-fill');

  let current = 0;
  const step = Math.max(1, Math.round(score / 30));
  const timer = setInterval(() => {
    current = Math.min(score, current + step);
    numEl.textContent = current;
    if (current >= score) clearInterval(timer);
  }, 25);

  const total = 515.22;
  const offset = total - (score / 100) * total;
  ringFill.style.strokeDashoffset = offset;

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
    badge.textContent = '🤖 AI + BG Removal';
  } else if (modeType === 'precision') {
    badge.classList.add('badge-precision');
    badge.textContent = '✋ Precision Mode';
  } else {
    badge.classList.add('badge-cv');
    badge.textContent = '✂️ CV BG Removal';
  }
}

/* ═══════════════════════════════════════════════════════
   8. MAIN ANALYSIS DISPATCHER
   ═══════════════════════════════════════════════════════ */

async function startAnalysis(img) {
  STATE.rawImage = img;
  showLoading();
  resetSteps();

  try {
    // Step 1: Read and scale image
    activateStep(0);
    await sleep(40);

    const maxDim = CONFIG.maxProcessDim;
    const W = img.naturalWidth || img.width;
    const H = img.naturalHeight || img.height;
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

    const resultCanvas = document.getElementById('result-canvas');
    resultCanvas.width = targetW;
    resultCanvas.height = targetH;

    if (STATE.mode === 'precision') {
      initPrecisionMode(resultCanvas);
      return;
    }

    // Step 2: Background Removal
    activateStep(1);
    await sleep(40);

    let aiResult = null;
    let aiBbox = null;

    if (STATE.apiKey) {
      try {
        aiResult = await callGeminiVision(procCanvas);
        if (aiResult.boundingBox) {
          aiBbox = {
            xmin: Math.max(0, Math.floor((aiResult.boundingBox.xmin / 1000) * targetW)),
            ymin: Math.max(0, Math.floor((aiResult.boundingBox.ymin / 1000) * targetH)),
            xmax: Math.min(targetW - 1, Math.ceil((aiResult.boundingBox.xmax / 1000) * targetW)),
            ymax: Math.min(targetH - 1, Math.ceil((aiResult.boundingBox.ymax / 1000) * targetH))
          };
        }
      } catch (aiErr) {
        console.warn('[AI] Vision API failed, continuing with autonomous CV segmentation:', aiErr.message);
      }
    }

    // Segment & Remove Background (HSV + Border Color Modeling + Connected Component)
    const bgRemoval = removeBackground(procCanvas, aiBbox);

    // Step 3: Corner Detection
    activateStep(2);
    await sleep(40);

    let corners = null;
    let rawAiCorners = null;

    if (aiResult && aiResult.corners) {
      const rawApex = { x: (aiResult.corners.apex.x / 1000) * targetW, y: (aiResult.corners.apex.y / 1000) * targetH };
      const rawBL   = { x: (aiResult.corners.bottomLeft.x / 1000) * targetW, y: (aiResult.corners.bottomLeft.y / 1000) * targetH };
      const rawBR   = { x: (aiResult.corners.bottomRight.x / 1000) * targetW, y: (aiResult.corners.bottomRight.y / 1000) * targetH };
      rawAiCorners = [rawApex, rawBL, rawBR];

      // Refine AI corners using the isolated samosa silhouette
      if (bgRemoval.hullPts && bgRemoval.hullPts.length >= 3) {
        const rad = Math.max(16, targetW * 0.12);
        corners = [
          findSharpestHullPoint(rawApex, bgRemoval.hullPts, rad, 'top'),
          findSharpestHullPoint(rawBL, bgRemoval.hullPts, rad, 'bottomLeft'),
          findSharpestHullPoint(rawBR, bgRemoval.hullPts, rad, 'bottomRight')
        ];
      } else {
        corners = rawAiCorners;
      }
    } else {
      // Autonomous pure-CV corner detection on isolated samosa silhouette
      corners = detectCornersFromSamosa(bgRemoval.hullPts, targetW, targetH);
    }

    // Step 4: Mathematical Calculation & Scoring
    activateStep(3);
    await sleep(40);

    const geom = computeGeometryResults(corners, bgRemoval.mask, bgRemoval.hullPts, targetW, targetH);

    const analysisResult = {
      refinedCorners: corners,
      rawAiCorners: rawAiCorners,
      hullPts: bgRemoval.hullPts,
      mask: bgRemoval.mask,
      bbox: aiBbox,
      isolatedCanvas: bgRemoval.isolatedCanvas,
      maskCanvas: bgRemoval.maskCanvas,
      geometry: geom,
      confidence: aiResult?.confidence ?? (bgRemoval.hasCleanSegmentation ? 0.92 : 0.65),
      modeType: aiResult ? 'ai' : 'cv'
    };

    STATE.lastResult = analysisResult;
    STATE.pipelineStage = 4; // default to final accuracy stage
    STATE.viewMode = 'nobg';  // default to isolated samosa view

    updateViewButtonsUI();
    updatePipelineStepperUI(4);

    renderOverlay(resultCanvas, analysisResult, STATE.viewMode, STATE.pipelineStage, STATE.debugMode);
    renderScore(geom.accuracyScore);
    renderMeasurements(geom);
    updateConfidenceUI(analysisResult.confidence, analysisResult.modeType === 'ai');
    updateDetectionBadge(analysisResult.modeType);

    showResults();

  } catch (err) {
    console.error('Analysis error:', err);
    showError('Analysis failed: ' + (err.message || String(err)));
  }
}

/* ═══════════════════════════════════════════════════════
   9. PRECISION MODE (MANUAL CORNER SELECTION)
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

  for (let i = 0; i < 3; i++) {
    const dot = document.getElementById(`prec-dot-${i}`);
    if (i < count) dot.classList.add('active');
    else dot.classList.remove('active');
  }

  overlay.querySelectorAll('.corner-pin').forEach(pin => pin.remove());

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
    isolatedCanvas: null,
    maskCanvas: null,
    geometry: geom,
    confidence: 1.0,
    modeType: 'precision'
  };

  STATE.lastResult = analysisResult;
  renderOverlay(canvas, analysisResult, 'original', 4, STATE.debugMode);
  renderScore(geom.accuracyScore);
  renderMeasurements(geom);
  updateDetectionBadge('precision');
}

/* ═══════════════════════════════════════════════════════
   10. DOM WIRING & EVENT HANDLERS
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

// View Switcher Buttons
const viewNobgBtn     = document.getElementById('view-nobg-btn');
const viewOriginalBtn = document.getElementById('view-original-btn');
const viewMaskBtn     = document.getElementById('view-mask-btn');

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

// API Key Modal Handlers
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

apiKeyToggle.addEventListener('click', () => {
  apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
});

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

// View Mode Switching Handlers (No BG, Original, Mask)
viewNobgBtn.addEventListener('click', () => setViewMode('nobg'));
viewOriginalBtn.addEventListener('click', () => setViewMode('original'));
viewMaskBtn.addEventListener('click', () => setViewMode('mask'));

function setViewMode(mode) {
  STATE.viewMode = mode;
  updateViewButtonsUI();
  if (STATE.lastResult && STATE.rawImage) {
    const canvas = document.getElementById('result-canvas');
    renderOverlay(canvas, STATE.lastResult, STATE.viewMode, STATE.pipelineStage, STATE.debugMode);
  }
}

function updateViewButtonsUI() {
  [
    { btn: viewNobgBtn, mode: 'nobg' },
    { btn: viewOriginalBtn, mode: 'original' },
    { btn: viewMaskBtn, mode: 'mask' }
  ].forEach(({ btn, mode }) => {
    if (STATE.viewMode === mode) {
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
    } else {
      btn.classList.remove('active');
      btn.setAttribute('aria-selected', 'false');
    }
  });

  const canvasWrap = document.getElementById('canvas-wrap');
  if (STATE.viewMode === 'nobg') {
    canvasWrap.classList.add('checkerboard-bg');
  } else {
    canvasWrap.classList.remove('checkerboard-bg');
  }
}

// Pipeline Stepper Handlers (Steps 1, 2, 3, 4 preview)
[1, 2, 3, 4].forEach(stageNum => {
  const card = document.getElementById(`step-card-${stageNum}`);
  if (card) {
    card.addEventListener('click', () => {
      STATE.pipelineStage = stageNum;
      updatePipelineStepperUI(stageNum);
      if (STATE.lastResult && STATE.rawImage) {
        const canvas = document.getElementById('result-canvas');
        renderOverlay(canvas, STATE.lastResult, STATE.viewMode, STATE.pipelineStage, STATE.debugMode);
      }
    });
  }
});

function updatePipelineStepperUI(activeStage) {
  [1, 2, 3, 4].forEach(num => {
    const card = document.getElementById(`step-card-${num}`);
    if (card) {
      if (num === activeStage) card.classList.add('active');
      else card.classList.remove('active');
    }
  });
}

// Debug Mode Toggle
debugToggle.addEventListener('change', (e) => {
  STATE.debugMode = e.target.checked;
  if (STATE.lastResult && STATE.rawImage) {
    renderOverlay(document.getElementById('result-canvas'), STATE.lastResult, STATE.viewMode, STATE.pipelineStage, STATE.debugMode);
  }
});

// Precision Overlay Click Handler
precisionOverlay.addEventListener('click', (e) => {
  if (STATE.precisionCorners.length >= 3) return;

  const canvas = document.getElementById('result-canvas');
  const rect = canvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  const canvasX = (clickX / rect.width) * canvas.width;
  const canvasY = (clickY / rect.height) * canvas.height;

  STATE.precisionCorners.push({ x: canvasX, y: canvasY });
  updatePrecisionUI();
});

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

// Test Difficult Samosa Button (resilient for both HTTP and file://)
sampleBtn.addEventListener('click', () => {
  showLoading();
  const img = new Image();
  img.onload = () => startAnalysis(img);
  img.onerror = async () => {
    try {
      const res = await fetch('test_samosa.jpg');
      if (!res.ok) throw new Error('Could not fetch sample.');
      const blob = await res.blob();
      img.src = URL.createObjectURL(blob);
    } catch (err) {
      showError('Sample photo error: ' + err.message);
    }
  };
  img.src = 'test_samosa.jpg';
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
