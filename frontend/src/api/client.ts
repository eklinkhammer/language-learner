import type {
  ChatMessage,
  ChatResponse,
  ExerciseEvalResponse,
  ExerciseListResponse,
  SpeechAnalysisResponse,
} from "../types";

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function analyzeSpeech(
  audio: Blob,
  expectedText: string,
  language = "es",
): Promise<SpeechAnalysisResponse> {
  const form = new FormData();
  form.append("audio", audio, "recording.webm");
  form.append("expected_text", expectedText);
  form.append("language", language);

  const res = await fetch(`${BASE}/speech/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function chat(
  message: string,
  language: string,
  history: ChatMessage[],
): Promise<ChatResponse> {
  return post("/tutor/chat", { message, language, history });
}

export async function getExercises(
  language = "es",
  difficulty?: string,
): Promise<ExerciseListResponse> {
  const params: Record<string, string> = { language };
  if (difficulty) params.difficulty = difficulty;
  return get("/exercises", params);
}

export async function evaluateExercise(
  audio: Blob,
  exerciseId: string,
  language = "es",
): Promise<ExerciseEvalResponse> {
  const form = new FormData();
  form.append("audio", audio, "recording.webm");
  form.append("exercise_id", exerciseId);
  form.append("language", language);

  const res = await fetch(`${BASE}/exercises/evaluate`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
