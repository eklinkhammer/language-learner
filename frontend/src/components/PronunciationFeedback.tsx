import { useCallback, useEffect, useMemo } from "react";
import type { FeedbackItem, PhonemeExampleRef, SpeechAnalysisResponse } from "../types";

interface Props {
  analysis: SpeechAnalysisResponse | null;
  userAudioBlob?: Blob | null;
  referenceAudioUrl?: string | null;
}

function PlayButton({ example }: { example: PhonemeExampleRef }) {
  const play = useCallback(() => {
    new Audio(example.tts_url).play();
  }, [example.tts_url]);

  return (
    <button className="play-btn" onClick={play} title={`Play "${example.example_word}"`}>
      &#9654; {example.example_word}
    </button>
  );
}

function StructuredFeedbackItem({ item }: { item: FeedbackItem }) {
  if (item.type === "summary" || item.type === "tip" || item.type === "silence_error") {
    return <li className="feedback-item">{item.message}</li>;
  }

  return (
    <li className="feedback-item feedback-item--phoneme">
      <span className="phoneme-pair">
        /{item.expected_phoneme}/ → you said /{item.actual_phoneme}/
      </span>
      <div className="phoneme-examples">
        {item.expected_example && (
          <span className="phoneme-example">
            <strong>Target:</strong> {item.expected_example.description}
            <PlayButton example={item.expected_example} />
          </span>
        )}
        {item.actual_example && (
          <span className="phoneme-example">
            <strong>Yours:</strong> {item.actual_example.description}
            <PlayButton example={item.actual_example} />
          </span>
        )}
        {!item.expected_example && !item.actual_example && (
          <span className="phoneme-example">{item.message}</span>
        )}
      </div>
    </li>
  );
}

export function PronunciationFeedback({ analysis, userAudioBlob, referenceAudioUrl }: Props) {
  const userAudioUrl = useMemo(
    () => (userAudioBlob ? URL.createObjectURL(userAudioBlob) : null),
    [userAudioBlob],
  );

  useEffect(() => {
    return () => {
      if (userAudioUrl) URL.revokeObjectURL(userAudioUrl);
    };
  }, [userAudioUrl]);

  if (!analysis) return null;

  const scorePercent = Math.round(analysis.overall_score * 100);
  const hasStructuredFeedback = analysis.feedback_items && analysis.feedback_items.length > 0;

  return (
    <div className="pronunciation-feedback">
      {(referenceAudioUrl || userAudioUrl) && (
        <div className="audio-comparison">
          {referenceAudioUrl && (
            <div className="audio-player">
              <label>Reference</label>
              <audio src={referenceAudioUrl} controls />
            </div>
          )}
          {userAudioUrl && (
            <div className="audio-player">
              <label>Your Recording</label>
              <audio src={userAudioUrl} controls />
            </div>
          )}
        </div>
      )}

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

      {hasStructuredFeedback ? (
        <ul className="feedback-list">
          {analysis.feedback_items!.map((item, i) => (
            <StructuredFeedbackItem key={i} item={item} />
          ))}
        </ul>
      ) : (
        <ul className="feedback-list">
          {analysis.feedback.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
