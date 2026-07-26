import { useEffect, useState } from "react";
import { getExercises, evaluateExercise } from "../api/client";
import { AudioRecorder } from "./AudioRecorder";
import { PronunciationFeedback } from "./PronunciationFeedback";
import type { Exercise, SpeechAnalysisResponse } from "../types";

type Difficulty = "" | "beginner" | "intermediate" | "advanced";

const DIFFICULTY_OPTIONS: { value: Difficulty; label: string }[] = [
  { value: "", label: "All" },
  { value: "beginner", label: "Beginner" },
  { value: "intermediate", label: "Intermediate" },
  { value: "advanced", label: "Advanced" },
];

interface Props {
  language: string;
}

export function ExerciseView({ language }: Props) {
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [current, setCurrent] = useState<Exercise | null>(null);
  const [analysis, setAnalysis] = useState<SpeechAnalysisResponse | null>(null);
  const [difficulty, setDifficulty] = useState<Difficulty>("");

  const [fetchLoading, setFetchLoading] = useState(true);
  const [evalLoading, setEvalLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  useEffect(() => {
    setFetchLoading(true);
    setFetchError(null);
    setCurrent(null);
    setAnalysis(null);
    getExercises(language, difficulty || undefined)
      .then((res) => {
        setExercises(res.exercises);
        if (res.exercises.length > 0) setCurrent(res.exercises[0]);
      })
      .catch(() => {
        setFetchError("Failed to load exercises. Is the backend running?");
      })
      .finally(() => setFetchLoading(false));
  }, [language, difficulty]);

  const handleRecording = async (blob: Blob) => {
    if (!current) return;
    setEvalLoading(true);
    setEvalError(null);
    setAnalysis(null);
    try {
      const res = await evaluateExercise(blob, current.id, current.language);
      setAnalysis(res.analysis);
    } catch {
      setEvalError("Pronunciation analysis failed. Please try again.");
    } finally {
      setEvalLoading(false);
    }
  };

  return (
    <div className="exercise-view">
      <h2>Pronunciation Drills</h2>

      <div className="difficulty-filter">
        {DIFFICULTY_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            className={difficulty === opt.value ? "active" : ""}
            onClick={() => setDifficulty(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {fetchError && <p className="error">{fetchError}</p>}

      {fetchLoading ? (
        <div className="loading-indicator">
          <div className="spinner" />
          <p className="loading-text">Loading exercises...</p>
        </div>
      ) : (
        <>
          <div className="exercise-list">
            {exercises.map((ex) => (
              <button
                key={ex.id}
                className={current?.id === ex.id ? "active" : ""}
                onClick={() => {
                  setCurrent(ex);
                  setAnalysis(null);
                  setEvalError(null);
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

              {evalLoading && (
                <div className="loading-indicator">
                  <div className="spinner" />
                  <p className="loading-text">Analyzing pronunciation...</p>
                </div>
              )}
              {evalError && <p className="error">{evalError}</p>}
              <PronunciationFeedback analysis={analysis} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
