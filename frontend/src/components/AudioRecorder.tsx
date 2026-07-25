import { useEffect, useMemo } from "react";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

interface Props {
  onRecordingComplete: (blob: Blob) => void;
}

function AudioPreview({ blob }: { blob: Blob }) {
  const url = useMemo(() => URL.createObjectURL(blob), [blob]);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  return <audio src={url} controls />;
}

export function AudioRecorder({ onRecordingComplete }: Props) {
  const { isRecording, audioBlob, error, startRecording, stopRecording, clearRecording } =
    useAudioRecorder();

  const handleStop = () => {
    stopRecording();
  };

  const handleSubmit = () => {
    if (audioBlob) {
      onRecordingComplete(audioBlob);
      clearRecording();
    }
  };

  return (
    <div className="audio-recorder">
      {error && <p className="error">{error}</p>}

      <div className="controls">
        {!isRecording && !audioBlob && (
          <button onClick={startRecording}>Record</button>
        )}
        {isRecording && (
          <button onClick={handleStop} className="recording">
            Stop
          </button>
        )}
        {audioBlob && (
          <>
            <AudioPreview blob={audioBlob} />
            <button onClick={handleSubmit}>Submit</button>
            <button onClick={clearRecording}>Discard</button>
          </>
        )}
      </div>
    </div>
  );
}
