import { useAudioRecorder } from "../hooks/useAudioRecorder";

interface Props {
  onRecordingComplete: (blob: Blob) => void;
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
            <audio src={URL.createObjectURL(audioBlob)} controls />
            <button onClick={handleSubmit}>Submit</button>
            <button onClick={clearRecording}>Discard</button>
          </>
        )}
      </div>
    </div>
  );
}
