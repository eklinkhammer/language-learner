export interface PhonemeScore {
  phoneme: string;
  expected: string;
  score: number;
  is_correct: boolean;
}

export interface SpeechAnalysisResponse {
  transcript: string;
  expected_text: string;
  phoneme_scores: PhonemeScore[];
  overall_score: number;
  feedback: string[];
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
