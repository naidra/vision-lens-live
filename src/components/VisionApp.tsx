import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  ObjectDetector,
  FaceLandmarker,
  GestureRecognizer,
  PoseLandmarker,
  type ObjectDetectorResult,
  type FaceLandmarkerResult,
  type GestureRecognizerResult,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Camera,
  Image as ImageIcon,
  Loader2,
  Sparkles,
  Sun,
  Moon,
  Hand,
  Smile,
  Boxes,
  ShieldCheck,
  Dumbbell,
  Activity,
  PersonStanding,
  RotateCcw,
} from "lucide-react";
import { useTheme } from "@/hooks/use-theme";

type Mode = "camera" | "image" | "video";
type Source = HTMLVideoElement | HTMLImageElement;
type RepPhase = "calibrating" | "down" | "up";
type GestureStat = { label: string; score: number; hand: string };

type RepSignalTracker = {
  phase: RepPhase;
  min: number | null;
  max: number | null;
  smoothed: number | null;
  samples: { value: number; t: number }[];
  lastRisingAt: number;
};

type RepTracker = {
  count: number;
  phase: RepPhase;
  height: RepSignalTracker;
  elbowFlex: RepSignalTracker;
  elbowExtend: RepSignalTracker;
  lastRepAt: number;
};

type PushUpTracker = {
  count: number;
  phase: RepPhase;
  elbowExtend: RepSignalTracker;
  lastRepAt: number;
};

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const VISIBLE_LANDMARK = 0.45;
const REP_LANDMARK_MIN_VISIBILITY = 0.18;
const REP_BODY_LANDMARK_MIN_VISIBILITY = 0.12;
const REP_HEIGHT_RANGE_MIN = 0.055;
const REP_ANGLE_RANGE_MIN = 0.12;
const REP_SAMPLE_WINDOW_MS = 4500;
const REP_SMOOTHING = 0.44;
const REP_OUTLIER_JUMP = 0.32;
const REP_DOWN_THRESHOLD = 0.32;
const REP_UP_THRESHOLD = 0.68;
const REP_COOLDOWN_MS = 420;
const REP_RISING_LOOKBACK_MS = 140;
const REP_RISING_DELTA_MIN = 0.003;
const REP_RISING_GRACE_MS = 260;
const PUSHUP_RANGE_MIN = 0.1;
const PUSHUP_POSTURE_MAX_VERTICAL_SPREAD = 0.36;
const PUSHUP_POSTURE_MIN_HORIZONTAL_SPREAD = 0.18;
const PUSHUP_COOLDOWN_MS = 500;
const GESTURE_TRACKING_OPTIONS = {
  numHands: 2,
  minHandDetectionConfidence: 0.3,
  minHandPresenceConfidence: 0.3,
  minTrackingConfidence: 0.3,
  cannedGesturesClassifierOptions: {
    scoreThreshold: 0.35,
  },
};

const getVisibleAverage = (landmarks: NormalizedLandmark[], indices: number[]) => {
  const points = indices
    .map((index) => landmarks[index])
    .filter((point) => point && point.visibility > VISIBLE_LANDMARK);

  if (!points.length) return null;

  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
};

const getAverageWithMinVisibility = (
  landmarks: NormalizedLandmark[],
  indices: number[],
  minVisibility: number,
) => {
  const points = indices
    .map((index) => landmarks[index])
    .filter((point) => point && (point.visibility ?? 0) >= minVisibility);

  if (!points.length) return null;

  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
};

const getRepPosition = (landmarks: NormalizedLandmark[]) => {
  const wrists = [landmarks[15], landmarks[16]].filter(Boolean);
  const visibleWrists = wrists.filter(
    (point) => (point.visibility ?? 0) >= REP_LANDMARK_MIN_VISIBILITY,
  );
  const elbows = [landmarks[13], landmarks[14]].filter(Boolean);
  const visibleElbows = elbows.filter(
    (point) => (point.visibility ?? 0) >= REP_LANDMARK_MIN_VISIBILITY,
  );
  const points = visibleWrists.length
    ? visibleWrists
    : visibleElbows.length
      ? visibleElbows
      : wrists;

  if (!points.length) return null;

  const totalWeight = points.reduce(
    (sum, point) => sum + Math.max(point.visibility ?? 0, REP_LANDMARK_MIN_VISIBILITY),
    0,
  );

  return {
    y:
      points.reduce(
        (sum, point) =>
          sum + point.y * Math.max(point.visibility ?? 0, REP_LANDMARK_MIN_VISIBILITY),
        0,
      ) / totalWeight,
  };
};

const createRepSignalTracker = (): RepSignalTracker => ({
  phase: "calibrating",
  min: null,
  max: null,
  smoothed: null,
  samples: [],
  lastRisingAt: 0,
});

const percentile = (values: number[], ratio: number) => {
  if (!values.length) return null;

  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * ratio)));
  return sorted[index];
};

const getAngle = (a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) => {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);

  if (!mag) return null;

  const cosine = Math.max(-1, Math.min(1, dot / mag));
  return (Math.acos(cosine) * 180) / Math.PI;
};

const getAverageElbowAngle = (landmarks: NormalizedLandmark[]) => {
  const sides = [
    [11, 13, 15],
    [12, 14, 16],
  ] as const;
  const angles = sides
    .map(([shoulderIndex, elbowIndex, wristIndex]) => {
      const shoulder = landmarks[shoulderIndex];
      const elbow = landmarks[elbowIndex];
      const wrist = landmarks[wristIndex];

      if (
        !shoulder ||
        !elbow ||
        !wrist ||
        (shoulder.visibility ?? 0) < REP_BODY_LANDMARK_MIN_VISIBILITY ||
        (elbow.visibility ?? 0) < REP_BODY_LANDMARK_MIN_VISIBILITY ||
        (wrist.visibility ?? 0) < REP_BODY_LANDMARK_MIN_VISIBILITY
      ) {
        return null;
      }

      return getAngle(shoulder, elbow, wrist);
    })
    .filter((angle): angle is number => angle != null);

  if (!angles.length) return null;

  return angles.reduce((sum, angle) => sum + angle, 0) / angles.length;
};

const getPushUpPosture = (landmarks: NormalizedLandmark[]) => {
  const keyIndices = [11, 12, 23, 24, 25, 26, 27, 28];
  const points = keyIndices
    .map((index) => landmarks[index])
    .filter((point) => point && (point.visibility ?? 0) >= REP_BODY_LANDMARK_MIN_VISIBILITY);
  const shoulders = getAverageWithMinVisibility(
    landmarks,
    [11, 12],
    REP_BODY_LANDMARK_MIN_VISIBILITY,
  );
  const hips = getAverageWithMinVisibility(landmarks, [23, 24], REP_BODY_LANDMARK_MIN_VISIBILITY);

  if (!shoulders || !hips || points.length < 4) return false;

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const horizontalSpread = Math.max(...xs) - Math.min(...xs);
  const verticalSpread = Math.max(...ys) - Math.min(...ys);
  const torsoVerticalGap = Math.abs(shoulders.y - hips.y);

  return (
    horizontalSpread >= PUSHUP_POSTURE_MIN_HORIZONTAL_SPREAD &&
    verticalSpread <= PUSHUP_POSTURE_MAX_VERTICAL_SPREAD &&
    torsoVerticalGap <= PUSHUP_POSTURE_MAX_VERTICAL_SPREAD * 0.55
  );
};

const updateRepSignal = (
  signal: RepSignalTracker,
  value: number | null,
  now: number,
  minRange: number,
) => {
  if (value == null) {
    return { ready: false, progress: 0, reachedTop: false, rising: false };
  }

  const lastValue = signal.smoothed ?? value;
  const rawValue = Math.abs(value - lastValue) > REP_OUTLIER_JUMP ? lastValue : value;
  const smoothed =
    signal.smoothed == null
      ? rawValue
      : signal.smoothed + (rawValue - signal.smoothed) * REP_SMOOTHING;

  signal.smoothed = smoothed;
  signal.samples = [...signal.samples, { value: smoothed, t: now }].filter(
    (sample) => now - sample.t <= REP_SAMPLE_WINDOW_MS,
  );
  const lookbackSample = [...signal.samples]
    .reverse()
    .find((sample) => now - sample.t >= REP_RISING_LOOKBACK_MS);
  const rising = lookbackSample
    ? smoothed - lookbackSample.value > REP_RISING_DELTA_MIN
    : smoothed - lastValue > REP_RISING_DELTA_MIN;
  if (rising) {
    signal.lastRisingAt = now;
  }

  const values = signal.samples.map((sample) => sample.value);
  const sampleMin = percentile(values, 0.08);
  const sampleMax = percentile(values, 0.92);

  if (sampleMin != null && sampleMax != null) {
    signal.min = signal.min == null ? sampleMin : signal.min * 0.8 + sampleMin * 0.2;
    signal.max = signal.max == null ? sampleMax : signal.max * 0.8 + sampleMax * 0.2;
  }

  const range = Math.max(0, (signal.max ?? smoothed) - (signal.min ?? smoothed));
  const ready = range > minRange;
  const progress = ready
    ? Math.max(0, Math.min(1, (smoothed - (signal.min ?? smoothed)) / range))
    : 0;

  if (!ready) {
    return { ready, progress, reachedTop: false, rising };
  }

  if (progress <= REP_DOWN_THRESHOLD) {
    signal.phase = "down";
  } else if (signal.phase === "calibrating") {
    signal.phase = progress > 0.5 ? "up" : "down";
  }

  return {
    ready,
    progress,
    rising,
    reachedTop: progress >= REP_UP_THRESHOLD && signal.phase === "down",
  };
};

const estimateRep = (poseRes: PoseLandmarkerResult | null, tracker: RepTracker) => {
  const pose = poseRes?.landmarks[0];
  if (!pose) {
    return {
      repCount: tracker.count,
      repPhase: tracker.phase,
      poseDetected: false,
      rangeReady: false,
      progress: 0,
    };
  }

  const now = performance.now();
  const wrists = getRepPosition(pose);
  const elbowAngle = getAverageElbowAngle(pose);
  const height = updateRepSignal(
    tracker.height,
    wrists ? 1 - wrists.y : null,
    now,
    REP_HEIGHT_RANGE_MIN,
  );
  const elbowFlex = updateRepSignal(
    tracker.elbowFlex,
    elbowAngle == null ? null : 1 - elbowAngle / 180,
    now,
    REP_ANGLE_RANGE_MIN,
  );
  const elbowExtend = updateRepSignal(
    tracker.elbowExtend,
    elbowAngle == null ? null : elbowAngle / 180,
    now,
    REP_ANGLE_RANGE_MIN,
  );

  const handIsRising = height.ready && now - tracker.height.lastRisingAt <= REP_RISING_GRACE_MS;
  const angleCanCount = handIsRising && height.progress >= 0.45 && tracker.height.phase === "down";
  const reachedTop =
    (height.reachedTop && handIsRising) ||
    (angleCanCount && (elbowFlex.reachedTop || elbowExtend.reachedTop));
  const ready = height.ready || elbowFlex.ready || elbowExtend.ready;
  const progress = height.ready
    ? height.progress
    : Math.max(elbowFlex.progress, elbowExtend.progress);

  if (reachedTop && now - tracker.lastRepAt > REP_COOLDOWN_MS) {
    tracker.count += 1;
    tracker.phase = "up";
    tracker.height.phase = "up";
    tracker.elbowFlex.phase = "up";
    tracker.elbowExtend.phase = "up";
    tracker.lastRepAt = now;
  } else if (
    [tracker.height, tracker.elbowFlex, tracker.elbowExtend].some(
      (signal) => signal.phase === "down",
    )
  ) {
    tracker.phase = "down";
  } else if (tracker.phase === "calibrating" && ready) {
    tracker.phase = progress > 0.5 ? "up" : "down";
  }

  return {
    repCount: tracker.count,
    repPhase: tracker.phase,
    poseDetected: true,
    rangeReady: ready,
    progress,
  };
};

const estimatePushUp = (poseRes: PoseLandmarkerResult | null, tracker: PushUpTracker) => {
  const pose = poseRes?.landmarks[0];
  if (!pose) {
    return {
      pushUpCount: tracker.count,
      pushUpPhase: tracker.phase,
      pushUpReady: false,
      pushUpProgress: 0,
    };
  }

  const now = performance.now();
  const inPushUpPosture = getPushUpPosture(pose);
  const elbowAngle = getAverageElbowAngle(pose);
  const elbowExtend = updateRepSignal(
    tracker.elbowExtend,
    inPushUpPosture && elbowAngle != null ? elbowAngle / 180 : null,
    now,
    PUSHUP_RANGE_MIN,
  );
  const rising = now - tracker.elbowExtend.lastRisingAt <= REP_RISING_GRACE_MS;
  const reachedTop = elbowExtend.reachedTop && rising;

  if (reachedTop && now - tracker.lastRepAt > PUSHUP_COOLDOWN_MS) {
    tracker.count += 1;
    tracker.phase = "up";
    tracker.elbowExtend.phase = "up";
    tracker.lastRepAt = now;
  } else if (elbowExtend.phase === "down") {
    tracker.phase = "down";
  } else if (tracker.phase === "calibrating" && elbowExtend.ready) {
    tracker.phase = elbowExtend.progress > 0.5 ? "up" : "down";
  }

  return {
    pushUpCount: tracker.count,
    pushUpPhase: tracker.phase,
    pushUpReady: inPushUpPosture && elbowExtend.ready,
    pushUpProgress: inPushUpPosture ? elbowExtend.progress : 0,
  };
};

export function VisionApp() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef<number>(-1);

  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const lastRepSoundCountRef = useRef(0);
  const lastPushUpSoundCountRef = useRef(0);
  const repTrackerRef = useRef<RepTracker>({
    count: 0,
    phase: "calibrating",
    height: createRepSignalTracker(),
    elbowFlex: createRepSignalTracker(),
    elbowExtend: createRepSignalTracker(),
    lastRepAt: 0,
  });
  const pushUpTrackerRef = useRef<PushUpTracker>({
    count: 0,
    phase: "calibrating",
    elbowExtend: createRepSignalTracker(),
    lastRepAt: 0,
  });

  const [loading, setLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState("Initializing WASM runtime…");
  const [mode, setMode] = useState<Mode>("camera");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [stats, setStats] = useState({
    objects: [] as { label: string; score: number }[],
    expression: null as string | null,
    gestures: [] as GestureStat[],
    repCount: 0,
    repPhase: "calibrating" as RepPhase,
    poseDetected: false,
    repRangeReady: false,
    repProgress: 0,
    pushUpCount: 0,
    pushUpPhase: "calibrating" as RepPhase,
    pushUpReady: false,
    pushUpProgress: 0,
    fps: 0,
  });
  const fpsRef = useRef({ frames: 0, t0: performance.now() });

  // Load all three models once on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoadProgress("Loading WASM runtime…");
        const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
        if (cancelled) return;

        setLoadProgress("Loading object detector…");
        const objectDetector = await ObjectDetector.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
            delegate: "GPU",
          },
          scoreThreshold: 0.45,
          runningMode: "VIDEO",
        });

        setLoadProgress("Loading face landmarker…");
        const faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          outputFaceBlendshapes: true,
          numFaces: 2,
        });

        setLoadProgress("Loading gesture recognizer…");
        const gestureRecognizer = await GestureRecognizer.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          ...GESTURE_TRACKING_OPTIONS,
        });

        setLoadProgress("Loading pose landmarker…");
        const poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.35,
          minPosePresenceConfidence: 0.35,
          minTrackingConfidence: 0.35,
        });

        if (cancelled) {
          objectDetector.close();
          faceLandmarker.close();
          gestureRecognizer.close();
          poseLandmarker.close();
          return;
        }

        objectDetectorRef.current = objectDetector;
        faceLandmarkerRef.current = faceLandmarker;
        gestureRecognizerRef.current = gestureRecognizer;
        poseLandmarkerRef.current = poseLandmarker;
        setLoading(false);
      } catch (err) {
        console.error(err);
        setLoadProgress(`Failed to load models: ${(err as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      objectDetectorRef.current?.close();
      faceLandmarkerRef.current?.close();
      gestureRecognizerRef.current?.close();
      poseLandmarkerRef.current?.close();
      audioContextRef.current?.close();
    };
  }, []);

  const stopLoop = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const stopCameraStream = useCallback(() => {
    const v = videoRef.current;
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
  }, []);

  const resetRepTracker = useCallback(() => {
    lastRepSoundCountRef.current = 0;
    lastPushUpSoundCountRef.current = 0;
    repTrackerRef.current = {
      count: 0,
      phase: "calibrating",
      height: createRepSignalTracker(),
      elbowFlex: createRepSignalTracker(),
      elbowExtend: createRepSignalTracker(),
      lastRepAt: 0,
    };
    pushUpTrackerRef.current = {
      count: 0,
      phase: "calibrating",
      elbowExtend: createRepSignalTracker(),
      lastRepAt: 0,
    };
    setStats((current) => ({
      ...current,
      repCount: 0,
      repPhase: "calibrating",
      repRangeReady: false,
      repProgress: 0,
      pushUpCount: 0,
      pushUpPhase: "calibrating",
      pushUpReady: false,
      pushUpProgress: 0,
    }));
  }, []);

  const playRepSound = useCallback(() => {
    const AudioContextConstructor =
      window.AudioContext ??
      (window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;

    if (!AudioContextConstructor) return;

    const audioContext = audioContextRef.current ?? new AudioContextConstructor();
    audioContextRef.current = audioContext;

    if (audioContext.state === "suspended") {
      void audioContext.resume();
    }

    const now = audioContext.currentTime;
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();

    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(620, now);
    oscillator.frequency.exponentialRampToValueAtTime(440, now + 0.12);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.315, now + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);

    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.15);
  }, []);

  const draw = useCallback(
    (
      source: Source,
      sw: number,
      sh: number,
      objRes: ObjectDetectorResult | null,
      faceRes: FaceLandmarkerResult | null,
      gestRes: GestureRecognizerResult | null,
      poseRes: PoseLandmarkerResult | null,
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      // Match canvas to source intrinsic size for crisp drawing
      if (canvas.width !== sw || canvas.height !== sh) {
        canvas.width = sw;
        canvas.height = sh;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, sw, sh);

      // Object boxes
      const objects: { label: string; score: number }[] = [];
      if (objRes) {
        for (const det of objRes.detections) {
          const box = det.boundingBox;
          const cat = det.categories[0];
          if (!box || !cat) continue;
          objects.push({ label: cat.categoryName, score: cat.score });
          ctx.strokeStyle = "oklch(0.78 0.18 200)";
          ctx.lineWidth = Math.max(2, sw / 400);
          ctx.strokeRect(box.originX, box.originY, box.width, box.height);
          const label = `${cat.categoryName} ${(cat.score * 100).toFixed(0)}%`;
          ctx.font = `${Math.max(14, sw / 60)}px ui-sans-serif, system-ui`;
          const pad = 4;
          const tw = ctx.measureText(label).width + pad * 2;
          const th = Math.max(18, sw / 50);
          ctx.fillStyle = "oklch(0.78 0.18 200)";
          ctx.fillRect(box.originX, Math.max(0, box.originY - th), tw, th);
          ctx.fillStyle = "oklch(0.15 0.02 260)";
          ctx.fillText(label, box.originX + pad, Math.max(th - 5, box.originY - 5));
        }
      }

      // Face landmarks (light dots) + dominant expression
      let expression: string | null = null;
      if (faceRes && faceRes.faceLandmarks.length > 0) {
        ctx.fillStyle = "oklch(0.85 0.15 330 / 0.7)";
        for (const lm of faceRes.faceLandmarks) {
          for (let i = 0; i < lm.length; i += 3) {
            const p = lm[i];
            ctx.fillRect(p.x * sw - 1, p.y * sh - 1, 2, 2);
          }
        }
        const blends = faceRes.faceBlendshapes?.[0]?.categories;
        if (blends && blends.length) {
          // Map specific blendshapes to friendly expressions
          const get = (name: string) => blends.find((b) => b.categoryName === name)?.score ?? 0;
          const candidates: { name: string; score: number }[] = [
            { name: "Smiling", score: (get("mouthSmileLeft") + get("mouthSmileRight")) / 2 },
            { name: "Frowning", score: (get("mouthFrownLeft") + get("mouthFrownRight")) / 2 },
            {
              name: "Surprised",
              score: (get("jawOpen") + get("eyeWideLeft") + get("eyeWideRight")) / 3,
            },
            { name: "Eyes closed", score: (get("eyeBlinkLeft") + get("eyeBlinkRight")) / 2 },
            { name: "Brows raised", score: get("browInnerUp") },
            { name: "Squinting", score: (get("eyeSquintLeft") + get("eyeSquintRight")) / 2 },
          ];
          candidates.sort((a, b) => b.score - a.score);
          if (candidates[0].score > 0.25) expression = candidates[0].name;
          else expression = "Neutral";
        }
      }

      // Hand landmarks + gestures
      const gestures: GestureStat[] = [];
      if (gestRes) {
        ctx.fillStyle = "oklch(0.82 0.18 145)";
        for (const hand of gestRes.landmarks) {
          for (const p of hand) {
            ctx.beginPath();
            ctx.arc(p.x * sw, p.y * sh, Math.max(3, sw / 220), 0, Math.PI * 2);
            ctx.fill();
          }
        }
        for (let i = 0; i < gestRes.gestures.length; i++) {
          const g = gestRes.gestures[i];
          const top = g[0];
          const handedness = gestRes.handedness[i]?.[0]?.categoryName ?? `Hand ${i + 1}`;
          gestures.push({
            hand: handedness,
            label:
              top?.categoryName === "None" ? "No gesture" : (top?.categoryName ?? "No gesture"),
            score: top?.score ?? 0,
          });
        }
      }

      if (poseRes && poseRes.landmarks.length > 0) {
        ctx.strokeStyle = "oklch(0.76 0.17 75)";
        ctx.lineWidth = Math.max(3, sw / 320);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        for (const pose of poseRes.landmarks) {
          for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
            const start = pose[connection.start];
            const end = pose[connection.end];
            if (
              !start ||
              !end ||
              start.visibility < VISIBLE_LANDMARK ||
              end.visibility < VISIBLE_LANDMARK
            ) {
              continue;
            }
            ctx.beginPath();
            ctx.moveTo(start.x * sw, start.y * sh);
            ctx.lineTo(end.x * sw, end.y * sh);
            ctx.stroke();
          }

          ctx.fillStyle = "oklch(0.82 0.18 75)";
          for (const point of pose) {
            if (point.visibility < VISIBLE_LANDMARK) continue;
            ctx.beginPath();
            ctx.arc(point.x * sw, point.y * sh, Math.max(3, sw / 240), 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      const reps = estimateRep(poseRes, repTrackerRef.current);
      if (reps.repCount > lastRepSoundCountRef.current) {
        lastRepSoundCountRef.current = reps.repCount;
        playRepSound();
      }
      const pushUps = estimatePushUp(poseRes, pushUpTrackerRef.current);
      if (pushUps.pushUpCount > lastPushUpSoundCountRef.current) {
        lastPushUpSoundCountRef.current = pushUps.pushUpCount;
        playRepSound();
      }

      // FPS
      const f = fpsRef.current;
      f.frames++;
      const now = performance.now();
      let fps = stats.fps;
      if (now - f.t0 > 500) {
        fps = Math.round((f.frames * 1000) / (now - f.t0));
        f.frames = 0;
        f.t0 = now;
      }

      setStats({
        objects: objects.slice(0, 8),
        expression,
        gestures,
        repCount: reps.repCount,
        repPhase: reps.repPhase,
        poseDetected: reps.poseDetected,
        repRangeReady: reps.rangeReady,
        repProgress: reps.progress,
        pushUpCount: pushUps.pushUpCount,
        pushUpPhase: pushUps.pushUpPhase,
        pushUpReady: pushUps.pushUpReady,
        pushUpProgress: pushUps.pushUpProgress,
        fps,
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [playRepSound],
  );

  const runVideoLoop = useCallback(() => {
    const video = videoRef.current;
    const od = objectDetectorRef.current;
    const fl = faceLandmarkerRef.current;
    const gr = gestureRecognizerRef.current;
    const pl = poseLandmarkerRef.current;
    if (!video || !od || !fl || !gr || !pl) return;

    const tick = () => {
      if (video.readyState >= 2 && !video.paused && !video.ended) {
        const ts = performance.now();
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const objRes = od.detectForVideo(video, ts);
          const faceRes = fl.detectForVideo(video, ts);
          const gestRes = gr.recognizeForVideo(video, ts);
          const poseRes = pl.detectForVideo(video, ts);
          draw(video, video.videoWidth, video.videoHeight, objRes, faceRes, gestRes, poseRes);
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [draw]);

  // Start camera
  const startCamera = useCallback(async () => {
    stopLoop();
    stopCameraStream();
    resetRepTracker();
    setMode("camera");
    setImageUrl(null);
    setVideoUrl(null);
    try {
      await objectDetectorRef.current?.setOptions({ runningMode: "VIDEO" });
      await faceLandmarkerRef.current?.setOptions({ runningMode: "VIDEO" });
      await gestureRecognizerRef.current?.setOptions({
        runningMode: "VIDEO",
        ...GESTURE_TRACKING_OPTIONS,
      });
      await poseLandmarkerRef.current?.setOptions({ runningMode: "VIDEO" });
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const v = videoRef.current!;
      v.srcObject = stream;
      v.muted = true;
      await v.play();
      runVideoLoop();
    } catch (err) {
      console.error("Camera error", err);
      alert("Could not access camera: " + (err as Error).message);
    }
  }, [resetRepTracker, runVideoLoop, stopCameraStream, stopLoop]);

  // Auto-start camera when models ready
  useEffect(() => {
    if (!loading && mode === "camera" && !videoRef.current?.srcObject) {
      startCamera();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  const handleFile = useCallback(
    async (file: File) => {
      stopLoop();
      stopCameraStream();
      resetRepTracker();
      const url = URL.createObjectURL(file);
      if (file.type.startsWith("image/")) {
        setMode("image");
        setVideoUrl(null);
        setImageUrl(url);
        // wait for img to load, then run once. Models need IMAGE running mode for images.
        await Promise.resolve();
        const img = imageRef.current!;
        await new Promise<void>((res) => {
          if (img.complete && img.naturalWidth) res();
          else img.onload = () => res();
        });
        const od = objectDetectorRef.current!;
        const fl = faceLandmarkerRef.current!;
        const gr = gestureRecognizerRef.current!;
        const pl = poseLandmarkerRef.current!;
        await od.setOptions({ runningMode: "IMAGE" });
        await fl.setOptions({ runningMode: "IMAGE" });
        await gr.setOptions({
          runningMode: "IMAGE",
          ...GESTURE_TRACKING_OPTIONS,
        });
        await pl.setOptions({ runningMode: "IMAGE" });
        const objRes = od.detect(img);
        const faceRes = fl.detect(img);
        const gestRes = gr.recognize(img);
        const poseRes = pl.detect(img);
        draw(img, img.naturalWidth, img.naturalHeight, objRes, faceRes, gestRes, poseRes);
      } else if (file.type.startsWith("video/")) {
        setMode("video");
        setImageUrl(null);
        setVideoUrl(url);
        const od = objectDetectorRef.current!;
        const fl = faceLandmarkerRef.current!;
        const gr = gestureRecognizerRef.current!;
        const pl = poseLandmarkerRef.current!;
        await od.setOptions({ runningMode: "VIDEO" });
        await fl.setOptions({ runningMode: "VIDEO" });
        await gr.setOptions({
          runningMode: "VIDEO",
          ...GESTURE_TRACKING_OPTIONS,
        });
        await pl.setOptions({ runningMode: "VIDEO" });
        const v = videoRef.current!;
        v.srcObject = null;
        v.src = url;
        v.muted = true;
        v.loop = true;
        await v.play();
        runVideoLoop();
      }
    },
    [draw, resetRepTracker, runVideoLoop, stopCameraStream, stopLoop],
  );

  const { theme, toggle } = useTheme();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border sticky top-0 z-10 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 min-w-0">
            <div className="h-8 w-8 rounded-md bg-primary/10 text-primary flex items-center justify-center shrink-0">
              <Sparkles className="h-4 w-4" />
            </div>
            <h1 className="text-sm font-semibold tracking-tight truncate">BrowserVision</h1>
            <Badge
              variant="secondary"
              className="ml-1 hidden md:inline-flex text-[10px] font-normal"
            >
              <ShieldCheck className="h-3 w-3 mr-1" /> 100% on-device · WASM
            </Badge>
          </div>
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant={mode === "camera" ? "default" : "ghost"}
              onClick={startCamera}
              disabled={loading}
              className="h-8"
            >
              <Camera className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Camera</span>
            </Button>
            <Button
              size="sm"
              variant={mode !== "camera" ? "default" : "ghost"}
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="h-8"
            >
              <ImageIcon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Upload</span>
            </Button>
            <div className="w-px h-5 bg-border mx-1" />
            <Button
              size="icon"
              variant="ghost"
              onClick={toggle}
              className="h-8 w-8"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? (
                <Sun className="h-3.5 w-3.5" />
              ) : (
                <Moon className="h-3.5 w-3.5" />
              )}
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
                e.target.value = "";
              }}
            />
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-4 grid lg:grid-cols-[1fr_300px] gap-4">
        <Card className="relative overflow-hidden bg-muted/30 border-border aspect-video flex items-center justify-center p-0">
          {loading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-background/90 backdrop-blur">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <p className="text-xs text-muted-foreground">{loadProgress}</p>
            </div>
          )}

          {mode === "image" && imageUrl ? (
            <img
              ref={imageRef}
              src={imageUrl}
              alt="Uploaded"
              className="max-h-full max-w-full object-contain"
              crossOrigin="anonymous"
            />
          ) : (
            <video
              ref={videoRef}
              src={videoUrl ?? undefined}
              className="max-h-full max-w-full object-contain"
              playsInline
              muted
              controls={mode === "video"}
            />
          )}

          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none object-contain"
          />

          {!loading && (
            <div className="absolute top-2.5 left-2.5 flex gap-1.5 text-[10px]">
              <Badge
                variant="secondary"
                className="font-mono tabular-nums backdrop-blur bg-background/70"
              >
                {stats.fps} FPS
              </Badge>
              <Badge variant="secondary" className="capitalize backdrop-blur bg-background/70">
                {mode}
              </Badge>
              <Badge
                variant="secondary"
                className="font-mono tabular-nums backdrop-blur bg-background/70"
              >
                {stats.repCount} reps
              </Badge>
              <Badge
                variant="secondary"
                className="font-mono tabular-nums backdrop-blur bg-background/70"
              >
                {stats.pushUpCount} push-ups
              </Badge>
            </div>
          )}
        </Card>

        <aside className="space-y-3">
          <Card className="p-3">
            <div className="flex items-start justify-between gap-2">
              <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
                <Dumbbell className="h-3 w-3" /> Barbell reps
              </h2>
              <Button
                size="icon"
                variant="ghost"
                onClick={resetRepTracker}
                className="h-7 w-7 -mt-1 -mr-1"
                aria-label="Reset rep counter"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
            <div className="flex items-end justify-between gap-3">
              <p className="text-4xl font-semibold tabular-nums leading-none">{stats.repCount}</p>
              <Badge
                variant={stats.repRangeReady ? "default" : "secondary"}
                className="capitalize text-[10px] font-normal"
              >
                {stats.poseDetected ? stats.repPhase : "No pose"}
              </Badge>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-150"
                style={{ width: `${Math.round(stats.repProgress * 100)}%` }}
              />
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground leading-relaxed">
              {stats.repRangeReady
                ? "Move from the bottom position to the top position to count one rep."
                : "Do one full warm-up rep so the tracker can learn your range."}
            </p>
          </Card>

          <Card className="p-3">
            <div className="flex items-start justify-between gap-2">
              <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
                <Activity className="h-3 w-3" /> Push-ups
              </h2>
              <Button
                size="icon"
                variant="ghost"
                onClick={resetRepTracker}
                className="h-7 w-7 -mt-1 -mr-1"
                aria-label="Reset exercise counters"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
            <div className="flex items-end justify-between gap-3">
              <p className="text-4xl font-semibold tabular-nums leading-none">
                {stats.pushUpCount}
              </p>
              <Badge
                variant={stats.pushUpReady ? "default" : "secondary"}
                className="capitalize text-[10px] font-normal"
              >
                {stats.poseDetected ? stats.pushUpPhase : "No pose"}
              </Badge>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-150"
                style={{ width: `${Math.round(stats.pushUpProgress * 100)}%` }}
              />
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground leading-relaxed">
              {stats.pushUpReady
                ? "Lower into the push-up, then press back up to count one rep."
                : "Get into a side-view push-up position so the tracker can learn your range."}
            </p>
          </Card>

          <Card className="p-3">
            <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-1.5 flex items-center gap-1.5">
              <PersonStanding className="h-3 w-3" /> Pose
            </h2>
            <p className="text-xl font-semibold text-foreground">
              {stats.poseDetected ? "Detected" : "—"}
            </p>
          </Card>

          <Card className="p-3">
            <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
              <Boxes className="h-3 w-3" /> Objects
            </h2>
            {stats.objects.length === 0 ? (
              <p className="text-xs text-muted-foreground">None detected</p>
            ) : (
              <div className="flex flex-wrap gap-1">
                {stats.objects.map((o, i) => (
                  <Badge
                    key={i}
                    variant="default"
                    className="capitalize text-[10px] font-normal py-0 px-1.5 h-5"
                  >
                    {o.label} <span className="opacity-60 ml-1">{(o.score * 100).toFixed(0)}%</span>
                  </Badge>
                ))}
              </div>
            )}
          </Card>

          <Card className="p-3">
            <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-1.5 flex items-center gap-1.5">
              <Smile className="h-3 w-3" /> Expression
            </h2>
            <p className="text-xl font-semibold text-foreground">{stats.expression ?? "—"}</p>
          </Card>

          <Card className="p-3">
            <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
              <Hand className="h-3 w-3" /> Gestures
            </h2>
            {stats.gestures.length === 0 ? (
              <p className="text-xs text-muted-foreground">No hands detected</p>
            ) : (
              <div className="flex flex-wrap gap-1">
                {stats.gestures.map((g, i) => (
                  <Badge
                    key={i}
                    variant="secondary"
                    className="text-[10px] font-normal py-0 px-1.5 h-5"
                  >
                    {g.hand}: {g.label}{" "}
                    <span className="opacity-60 ml-1">{(g.score * 100).toFixed(0)}%</span>
                  </Badge>
                ))}
              </div>
            )}
          </Card>

          <Card className="p-3 text-[11px] text-muted-foreground leading-relaxed">
            <p className="text-foreground font-medium mb-1">How it works</p>
            EfficientDet-Lite, FaceLandmarker, GestureRecognizer and PoseLandmarker run via
            MediaPipe Tasks WebAssembly with GPU acceleration. Reps are counted from wrist height
            cycles. No data leaves your device.
          </Card>
        </aside>
      </main>
    </div>
  );
}
