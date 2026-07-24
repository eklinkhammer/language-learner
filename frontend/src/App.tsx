import { useState } from "react";
import { ConversationView } from "./components/ConversationView";
import { ExerciseView } from "./components/ExerciseView";

type Mode = "conversation" | "exercises";

function App() {
  const [mode, setMode] = useState<Mode>("exercises");

  return (
    <>
      <header>
        <h1>Language Learner</h1>
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
        {mode === "conversation" ? <ConversationView /> : <ExerciseView />}
      </main>
    </>
  );
}

export default App;
