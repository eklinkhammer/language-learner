import type { SpeechAnalysisResponse } from "../types";

interface Props {
  analysis: SpeechAnalysisResponse | null;
}

export function PronunciationFeedback({ analysis }: Props) {
  if (!analysis) return null;

  const scorePercent = Math.round(analysis.overall_score * 100);

  return (
    <div className="pronunciation-feedback">
      <h3>
        Score: {scorePercent}%
      </h3>

      <p>
        <strong>You said:</strong> {analysis.transcript}
      </p>
      <p>
        <strong>Expected:</strong> {analysis.expected_text}
      </p>

      <div className="phoneme-scores">
        {analysis.phoneme_scores.map((ps, i) => (
          <span
            key={i}
            className={ps.is_correct ? "correct" : "incorrect"}
            title={`Expected: ${ps.expected}, Score: ${Math.round(ps.score * 100)}%`}
          >
            {ps.phoneme}
          </span>
        ))}
      </div>

      <ul className="feedback-list">
        {analysis.feedback.map((f, i) => (
          <li key={i}>{f}</li>
        ))}
      </ul>
    </div>
  );
}
