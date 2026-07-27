export interface PhonemeScore {
  phoneme: string;
  expected: string;
  score: number;
  is_correct: boolean;
}

export interface PhonemeExampleRef {
  phoneme: string;
  example_word: string;
  highlight: string;
  description: string;
  tts_url: string;
}

export interface FeedbackItem {
  type: "summary" | "phoneme_error" | "silence_error" | "tip" | "word_breakdown";
  message: string;
  expected_phoneme?: string;
  actual_phoneme?: string;
  expected_example?: PhonemeExampleRef;
  actual_example?: PhonemeExampleRef;
  source_word?: string;
  source_letter?: string;
}

export interface SpeechAnalysisResponse {
  transcript: string;
  expected_text: string;
  phoneme_scores: PhonemeScore[];
  overall_score: number;
  feedback: string[];
  feedback_items?: FeedbackItem[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  suggestions: string[];
}

export interface Exercise {
  id: string;
  language: string;
  phrase: string;
  ipa: string;
  translation: string;
  difficulty: "beginner" | "intermediate" | "advanced";
}

export interface ExerciseListResponse {
  exercises: Exercise[];
}

export interface ExerciseEvalResponse {
  exercise: Exercise;
  analysis: SpeechAnalysisResponse;
}
