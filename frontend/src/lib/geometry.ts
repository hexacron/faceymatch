/**
 * Letterbox geometry for the detection overlay.
 *
 * The image element is styled `object-fit: contain`, so the bitmap is scaled
 * uniformly and centred inside the element's content box, leaving bars on one
 * axis. Detection boxes arrive in original-image pixels, so every draw and
 * every hit test has to go through the same mapping or the boxes land off the
 * faces. The canvas sits on top of the image at exactly the element rect and
 * is sized in device pixels, with the 2D context pre-scaled by
 * `devicePixelRatio` so the rest of the drawing code can work in CSS pixels.
 */

export type Rect = { x: number; y: number; w: number; h: number };

/**
 * Express a rect as fractions of a bitmap, so a box can be compared across two
 * different encodings of the same picture (a 1280-wide live frame and the
 * stored file the pipeline actually detected on).
 */
export function normalizeRect(rect: Rect, width: number, height: number): Rect {
  return { x: rect.x / width, y: rect.y / height, w: rect.w / width, h: rect.h / height };
}

/** The `object-fit: contain` placement of a bitmap inside an element box. */
export type Letterbox = {
  /** Uniform image-pixel -> CSS-pixel scale factor. Zero when unmeasurable. */
  scale: number;
  /** CSS-pixel offset of the drawn bitmap's top-left inside the element box. */
  offsetX: number;
  offsetY: number;
  /** Drawn bitmap size in CSS pixels. */
  width: number;
  height: number;
};

/**
 * Compute the contain-fit placement. A zero or unknown natural size (the image
 * has not decoded yet) yields a degenerate box with `scale === 0` so callers
 * skip drawing rather than divide by zero.
 */
export function containLetterbox(
  naturalWidth: number,
  naturalHeight: number,
  boxWidth: number,
  boxHeight: number,
): Letterbox {
  if (naturalWidth <= 0 || naturalHeight <= 0 || boxWidth <= 0 || boxHeight <= 0) {
    return { scale: 0, offsetX: 0, offsetY: 0, width: 0, height: 0 };
  }
  const scale = Math.min(boxWidth / naturalWidth, boxHeight / naturalHeight);
  const width = naturalWidth * scale;
  const height = naturalHeight * scale;
  return {
    scale,
    offsetX: (boxWidth - width) / 2,
    offsetY: (boxHeight - height) / 2,
    width,
    height,
  };
}

/** Map a rect in image pixels to CSS pixels relative to the element box. */
export function imageRectToCss(rect: Rect, box: Letterbox): Rect {
  return {
    x: box.offsetX + rect.x * box.scale,
    y: box.offsetY + rect.y * box.scale,
    w: rect.w * box.scale,
    h: rect.h * box.scale,
  };
}

/**
 * Hit-test a click, given in CSS pixels relative to the element box, against
 * rects in image pixels. Returns the index of the smallest containing rect, so
 * a face box nested inside a larger one stays clickable, or null for a miss.
 */
export function hitTestImageRects(
  rects: readonly Rect[],
  cssX: number,
  cssY: number,
  box: Letterbox,
): number | null {
  if (box.scale <= 0) {
    return null;
  }
  const x = (cssX - box.offsetX) / box.scale;
  const y = (cssY - box.offsetY) / box.scale;
  let best: number | null = null;
  let bestArea = Number.POSITIVE_INFINITY;
  for (let index = 0; index < rects.length; index += 1) {
    const rect = rects[index];
    if (rect === undefined) {
      continue;
    }
    if (x < rect.x || y < rect.y || x > rect.x + rect.w || y > rect.y + rect.h) {
      continue;
    }
    const area = rect.w * rect.h;
    if (area < bestArea) {
      bestArea = area;
      best = index;
    }
  }
  return best;
}

/**
 * Index of the rect that best overlaps `target` by intersection-over-union, or
 * null when nothing reaches `minIou`. Both sides must be in the same
 * coordinate space; the live handoff normalises to fractions of the frame so
 * the clicked box can be compared against stored samples in image pixels.
 *
 * IoU rather than centre distance: a detector run twice over the same frame
 * shifts a box by a few pixels, but a *different* face never reaches a
 * meaningful overlap, so a miss stays a miss instead of snapping to a
 * neighbour.
 */
export function bestOverlapRect(
  rects: readonly Rect[],
  target: Rect,
  minIou: number,
): number | null {
  let best: number | null = null;
  let bestIou = minIou;
  const targetArea = target.w * target.h;
  for (let index = 0; index < rects.length; index += 1) {
    const rect = rects[index];
    if (rect === undefined) {
      continue;
    }
    const overlapW = Math.min(rect.x + rect.w, target.x + target.w) - Math.max(rect.x, target.x);
    const overlapH = Math.min(rect.y + rect.h, target.y + target.h) - Math.max(rect.y, target.y);
    if (overlapW <= 0 || overlapH <= 0) {
      continue;
    }
    const intersection = overlapW * overlapH;
    const iou = intersection / (rect.w * rect.h + targetArea - intersection);
    if (iou > bestIou) {
      bestIou = iou;
      best = index;
    }
  }
  return best;
}

/**
 * Resize a canvas backing store for the current CSS size and device pixel
 * ratio and return a context whose user space is CSS pixels. Returns null when
 * the 2D context is unavailable (never in practice, but the DOM lib types it
 * as nullable and a non-null assertion would hide a real failure).
 */
export function prepareCanvas(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  dpr: number,
): CanvasRenderingContext2D | null {
  const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
  const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== pixelWidth) {
    canvas.width = pixelWidth;
  }
  if (canvas.height !== pixelHeight) {
    canvas.height = pixelHeight;
  }
  canvas.style.width = `${String(cssWidth)}px`;
  canvas.style.height = `${String(cssHeight)}px`;
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    return null;
  }
  // One unit of user space is one CSS pixel: the backing store stays at device
  // resolution (crisp hairlines and labels on a HiDPI screen) while the drawing
  // code keeps working in the same space as the layout and the pointer events.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
