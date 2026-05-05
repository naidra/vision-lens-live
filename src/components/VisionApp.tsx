import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  ObjectDetector,
  FaceLandmarker,
  GestureRecognizer,
  type ObjectDetectorResult,
  type FaceLandmarkerResult,
  type GestureRecognizerResult,
} from "@mediapipe/tasks-vision";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Camera, Image as ImageIcon, Video, Loader2, Sparkles } from "lucide-react";

type Mode = "camera" | "image" | "video";
type Source = HTMLVideoElement | HTMLImageElement;

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

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

  const [loading, setLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState("Initializing WASM runtime…");
  const [mode, setMode] = useState<Mode>("camera");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [stats, setStats] = useState({
    objects: [] as { label: string; score: number }[],
    expression: null as string | null,
    gestures: [] as { label: string; score: number }[],
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
          numHands: 2,
        });

        if (cancelled) {
          objectDetector.close();
          faceLandmarker.close();
          gestureRecognizer.close();
          return;
        }

        objectDetectorRef.current = objectDetector;
        faceLandmarkerRef.current = faceLandmarker;
        gestureRecognizerRef.current = gestureRecognizer;
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

  const draw = useCallback(
    (
      source: Source,
      sw: number,
      sh: number,
      objRes: ObjectDetectorResult | null,
      faceRes: FaceLandmarkerResult | null,
      gestRes: GestureRecognizerResult | null,
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
          const get = (name: string) =>
            blends.find((b) => b.categoryName === name)?.score ?? 0;
          const candidates: { name: string; score: number }[] = [
            { name: "Smiling", score: (get("mouthSmileLeft") + get("mouthSmileRight")) / 2 },
            { name: "Frowning", score: (get("mouthFrownLeft") + get("mouthFrownRight")) / 2 },
            { name: "Surprised", score: (get("jawOpen") + get("eyeWideLeft") + get("eyeWideRight")) / 3 },
            { name: "Eyes closed", score: (get("eyeBlinkLeft") + get("eyeBlinkRight")) / 2 },
            { name: "Brows raised", score: (get("browInnerUp")) },
            { name: "Squinting", score: (get("eyeSquintLeft") + get("eyeSquintRight")) / 2 },
          ];
          candidates.sort((a, b) => b.score - a.score);
          if (candidates[0].score > 0.25) expression = candidates[0].name;
          else expression = "Neutral";
        }
      }

      // Hand landmarks + gestures
      const gestures: { label: string; score: number }[] = [];
      if (gestRes) {
        ctx.fillStyle = "oklch(0.82 0.18 145)";
        for (const hand of gestRes.landmarks) {
          for (const p of hand) {
            ctx.beginPath();
            ctx.arc(p.x * sw, p.y * sh, Math.max(3, sw / 220), 0, Math.PI * 2);
            ctx.fill();
          }
        }
        for (const g of gestRes.gestures) {
          const top = g[0];
          if (top && top.categoryName !== "None") {
            gestures.push({ label: top.categoryName, score: top.score });
          }
        }
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

      setStats({ objects: objects.slice(0, 8), expression, gestures, fps });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const runVideoLoop = useCallback(() => {
    const video = videoRef.current;
    const od = objectDetectorRef.current;
    const fl = faceLandmarkerRef.current;
    const gr = gestureRecognizerRef.current;
    if (!video || !od || !fl || !gr) return;

    const tick = () => {
      if (video.readyState >= 2 && !video.paused && !video.ended) {
        const ts = performance.now();
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const objRes = od.detectForVideo(video, ts);
          const faceRes = fl.detectForVideo(video, ts);
          const gestRes = gr.recognizeForVideo(video, ts);
          draw(video, video.videoWidth, video.videoHeight, objRes, faceRes, gestRes);
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
    setMode("camera");
    setImageUrl(null);
    setVideoUrl(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const v = videoRef.current!;
      v.srcObject = stream;
      v.muted = true;
      await v.play();
      // Switch models to VIDEO mode (already are)
      runVideoLoop();
    } catch (err) {
      console.error("Camera error", err);
      alert("Could not access camera: " + (err as Error).message);
    }
  }, [runVideoLoop, stopCameraStream, stopLoop]);

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
        await od.setOptions({ runningMode: "IMAGE" });
        await fl.setOptions({ runningMode: "IMAGE" });
        await gr.setOptions({ runningMode: "IMAGE" });
        const objRes = od.detect(img);
        const faceRes = fl.detect(img);
        const gestRes = gr.recognize(img);
        draw(img, img.naturalWidth, img.naturalHeight, objRes, faceRes, gestRes);
      } else if (file.type.startsWith("video/")) {
        setMode("video");
        setImageUrl(null);
        setVideoUrl(url);
        const od = objectDetectorRef.current!;
        const fl = faceLandmarkerRef.current!;
        const gr = gestureRecognizerRef.current!;
        await od.setOptions({ runningMode: "VIDEO" });
        await fl.setOptions({ runningMode: "VIDEO" });
        await gr.setOptions({ runningMode: "VIDEO" });
        const v = videoRef.current!;
        v.srcObject = null;
        v.src = url;
        v.muted = true;
        v.loop = true;
        await v.play();
        runVideoLoop();
      }
    },
    [draw, runVideoLoop, stopCameraStream, stopLoop],
  );

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/60 backdrop-blur sticky top-0 z-10 bg-background/70">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <h1 className="text-lg font-semibold tracking-tight">
              BrowserVision
            </h1>
            <Badge variant="secondary" className="ml-2 hidden sm:inline-flex">
              100% in-browser · WASM
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant={mode === "camera" ? "default" : "secondary"}
              onClick={startCamera}
              disabled={loading}
            >
              <Camera className="h-4 w-4" /> Camera
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              <ImageIcon className="h-4 w-4" /> Image / Video
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

      <main className="max-w-6xl mx-auto px-4 py-6 grid lg:grid-cols-[1fr_320px] gap-6">
        <Card className="relative overflow-hidden bg-card border-border/60 aspect-video flex items-center justify-center">
          {loading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-background/80 backdrop-blur">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">{loadProgress}</p>
            </div>
          )}

          {/* Source layer */}
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

          {/* Overlay canvas */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none object-contain"
          />

          {!loading && (
            <div className="absolute top-3 left-3 flex gap-2 text-xs">
              <Badge variant="secondary">{stats.fps} FPS</Badge>
              <Badge variant="secondary" className="capitalize">
                {mode}
              </Badge>
            </div>
          )}
        </Card>

        <aside className="space-y-4">
          <Card className="p-4">
            <h2 className="text-sm font-medium mb-2 text-muted-foreground">
              Objects
            </h2>
            {stats.objects.length === 0 ? (
              <p className="text-xs text-muted-foreground">None detected</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {stats.objects.map((o, i) => (
                  <Badge key={i} variant="default" className="capitalize">
                    {o.label} · {(o.score * 100).toFixed(0)}%
                  </Badge>
                ))}
              </div>
            )}
          </Card>
          <Card className="p-4">
            <h2 className="text-sm font-medium mb-2 text-muted-foreground">
              Face expression
            </h2>
            <p className="text-2xl font-semibold">
              {stats.expression ?? "—"}
            </p>
          </Card>
          <Card className="p-4">
            <h2 className="text-sm font-medium mb-2 text-muted-foreground">
              Gestures
            </h2>
            {stats.gestures.length === 0 ? (
              <p className="text-xs text-muted-foreground">No hands detected</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {stats.gestures.map((g, i) => (
                  <Badge key={i} variant="secondary">
                    {g.label} · {(g.score * 100).toFixed(0)}%
                  </Badge>
                ))}
              </div>
            )}
          </Card>
          <Card className="p-4 text-xs text-muted-foreground leading-relaxed">
            <p className="flex items-center gap-1.5 mb-1 text-foreground font-medium">
              <Video className="h-3.5 w-3.5" /> How it works
            </p>
            Models (EfficientDet-Lite, FaceLandmarker, GestureRecognizer) run via
            MediaPipe Tasks WebAssembly with GPU delegate. No data ever leaves
            your device.
          </Card>
        </aside>
      </main>
    </div>
  );
}
