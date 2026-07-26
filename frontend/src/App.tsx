import { useState } from "react";
import { ConversationView } from "./components/ConversationView";
import { ExerciseView } from "./components/ExerciseView";

type Mode = "conversation" | "exercises";

const LANGUAGES = [
  { code: "es", label: "Spanish" },
  { code: "hr", label: "Croatian" },
  { code: "de", label: "German" },
  { code: "zh", label: "Chinese" },
] as const;

function App() {
  const [mode, setMode] = useState<Mode>("exercises");
  const [language, setLanguage] = useState("es");

  return (
    <>
      <header>
        <h1>Language Learner</h1>
        <div className="language-selector">
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
        <nav>
          <button
            className={mode === "conversation" ? "active" : ""}
            onClick={() => setMode("conversation")}
          >
            Conversation
          </button>
          <button
            className={mode === "exercises" ? "active" : ""}
            onClick={() => setMode("exercises")}
          >
            Pronunciation Drills
          </button>
        </nav>
      </header>

      <main>
        {mode === "conversation" ? (
          <ConversationView key={language} language={language} />
        ) : (
          <ExerciseView key={language} language={language} />
        )}
      </main>
    </>
  );
}

export default App;
