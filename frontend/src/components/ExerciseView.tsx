import { useEffect, useState } from "react";
import { getExercises, evaluateExercise } from "../api/client";
import { AudioRecorder } from "./AudioRecorder";
import { PronunciationFeedback } from "./PronunciationFeedback";
import type { Exercise, SpeechAnalysisResponse } from "../types";

export function ExerciseView() {
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [current, setCurrent] = useState<Exercise | null>(null);
  const [analysis, setAnalysis] = useState<SpeechAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getExercises("es")
      .then((res) => {
        setExercises(res.exercises);
        if (res.exercises.length > 0) setCurrent(res.exercises[0]);
      })
      .catch(() => {
        setError("Failed to load exercises. Is the backend running?");
      });
  }, []);

  const handleRecording = async (blob: Blob) => {
    if (!current) return;
    setLoading(true);
    setAnalysis(null);
    try {
      const res = await evaluateExercise(blob, current.id, current.language);
      setAnalysis(res.analysis);
    } catch {
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="exercise-view">
      <h2>Pronunciation Drills</h2>

      {error && <p className="error">{error}</p>}

      <div className="exercise-list">
        {exercises.map((ex) => (
          <button
            key={ex.id}
            className={current?.id === ex.id ? "active" : ""}
            onClick={() => {
              setCurrent(ex);
              setAnalysis(null);
            }}
          >
            {ex.phrase}
          </button>
        ))}
      </div>

      {current && (
        <div className="current-exercise">
          <p className="phrase">{current.phrase}</p>
          <p className="ipa">/{current.ipa}/</p>
          <p className="translation">{current.translation}</p>

          <AudioRecorder onRecordingComplete={handleRecording} />
          {loading && <p>Analyzing...</p>}
          <PronunciationFeedback analysis={analysis} />
        </div>
      )}
    </div>
  );
}
