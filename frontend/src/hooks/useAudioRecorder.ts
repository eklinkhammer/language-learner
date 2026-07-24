import { useCallback, useRef, useState } from "react";

export interface AudioRecorderState {
  isRecording: boolean;
  audioBlob: Blob | null;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  clearRecording: () => void;
}

export function useAudioRecorder(): AudioRecorderState {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setAudioBlob(null);
      chunks.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: recorder.mimeType });
        setAudioBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to access microphone");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorder.current && mediaRecorder.current.state === "recording") {
      mediaRecorder.current.stop();
      setIsRecording(false);
    }
  }, []);

  const clearRecording = useCallback(() => {
    setAudioBlob(null);
    setError(null);
  }, []);

  return { isRecording, audioBlob, error, startRecording, stopRecording, clearRecording };
}
