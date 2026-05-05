import { createFileRoute } from "@tanstack/react-router";
import { VisionApp } from "@/components/VisionApp";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "BrowserVision — In-Browser AI: Objects, Faces, Gestures" },
      {
        name: "description",
        content:
          "Real-time object detection, face & expression recognition, and gesture detection — running 100% in your browser via MediaPipe WASM.",
      },
    ],
  }),
});

function Index() {
  return <VisionApp />;
}
