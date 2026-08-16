import React, { useState, useEffect } from "react";
import Groq from "groq-sdk";

const Prediction = ({ prediction }) => {
  const grok_api = import.meta.env.VITE_GEN_AI;
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!prediction || !grok_api) {
      setResult(null);
      return;
    }

    const groq = new Groq({
      apiKey: grok_api,
      dangerouslyAllowBrowser: true,
    });

    const getAIResponse = async () => {
      const rawConfidence = prediction.confidence;
      const confidenceText =
        typeof rawConfidence === "number"
          ? `${rawConfidence.toFixed(2)}%`
          : typeof rawConfidence === "string"
          ? rawConfidence.includes("%")
            ? rawConfidence
            : `${rawConfidence}%`
          : "0.00%";

      const prompt = `
You are a medical assistant. Generate a short, clear HTML response only.

Rules:
- Use only semantic HTML tags such as <h3>, <p>, <ul>, <li>, <strong>.
- No markdown, no code blocks, no triple backticks, no plain text outside HTML.
- If the detected result is "notumor", respond with one cheerful congratulatory paragraph like: <p>Congratulations! Based on the prediction, you do not appear to have a tumor. Please continue routine health checkups.</p>
- If the detected result is any type of cancer or tumor, give only medically relevant information in this exact style:
  1. A short heading: <h3>Possible Condition</h3>
  2. A short explanation paragraph
  3. <h3>Symptoms</h3> with 4 bullet points
  4. <h3>Possible Causes</h3> with 3 bullet points
  5. <h3>Effects</h3> with 3 bullet points
  6. <h3>Probable Medical Advice</h3> with 3 bullet points
- Keep the answer concise, practical, and non-alarmist.
- Do not include unrelated cancer types, general knowledge, or extra advice outside this structure.

Prediction result: ${prediction.result}
Confidence: ${confidenceText}
`;

      try {
        setResult(null);
        const response = await groq.chat.completions.create({
          messages: [{ role: "user", content: prompt }],
          model: "llama-3.1-8b-instant",
        });

        const resultText =
          response.choices[0]?.message?.content ||
          "<p>Unable to generate an explanation for this prediction.</p>";

        setResult(resultText);
      } catch (error) {
        console.error("AI error:", error);
        setResult("<p>Oops! Something went wrong while generating the prediction.</p>");
      }
    };

    getAIResponse();
  }, [prediction, grok_api]);

  if (!prediction) {
    return (
      <div className="prediction-container empty-state">
        <h2>Prediction Result</h2>
        <p className="no-prediction">No prediction available yet.</p>
      </div>
    );
  }

  const rawConfidence = prediction.confidence;
  const parsedConfidence =
    typeof rawConfidence === "number"
      ? rawConfidence
      : typeof rawConfidence === "string"
      ? Number.parseFloat(rawConfidence.replace("%", ""))
      : 0;
  const confidence = Number.isFinite(parsedConfidence) ? parsedConfidence : 0;
  const resultLabel = prediction.result || "Not available";

  return (
    <div className="prediction-container">
      <div className="result-header">
        <div>
          <p className="panel-kicker">AI summary</p>
          <h2>Prediction Result</h2>
        </div>
        <span className={`confidence-badge ${confidence >= 70 ? "high" : confidence >= 40 ? "medium" : "low"}`}>
          {confidence.toFixed(2)}% confidence
        </span>
      </div>

      <div className="classification-box">
        <span className="label">Diagnosis</span>
        <strong>{resultLabel}</strong>
      </div>

      {result ? (
        <div className="result-html" dangerouslySetInnerHTML={{ __html: result }} />
      ) : (
        <div className="loading-box">
          <div className="spinner" />
          <p>Generating clinical summary...</p>
        </div>
      )}
    </div>
  );
};

export default Prediction;
