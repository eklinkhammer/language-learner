import { useState } from "react";
import { chat } from "../api/client";
import type { ChatMessage } from "../types";

export function ConversationView() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [language] = useState("es");

  const send = async () => {
    if (!input.trim()) return;

    const userMsg: ChatMessage = { role: "user", content: input };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setLoading(true);

    try {
      const res = await chat(input, language, messages);
      setMessages([...updated, { role: "assistant", content: res.reply }]);
    } catch {
      setMessages([
        ...updated,
        { role: "assistant", content: "Error: could not reach tutor." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="conversation-view">
      <h2>Conversation Tutor</h2>

      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <strong>{m.role === "user" ? "You" : "Tutor"}:</strong> {m.content}
          </div>
        ))}
        {loading && <div className="message assistant">Tutor is typing...</div>}
      </div>

      <div className="input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Type in Spanish..."
        />
        <button onClick={send} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}
