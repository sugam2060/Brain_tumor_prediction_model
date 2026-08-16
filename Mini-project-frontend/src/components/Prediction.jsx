import React, { useState, useEffect } from "react";
import Groq from "groq-sdk";

const Prediction = ({ prediction }) => {
  const grok_api = import.meta.env.VITE_GEN_AI;
  const [result, setResult] = useState(null);

  const rawConfidence = prediction?.confidence;
  const parsedConfidence =
    typeof rawConfidence === "number"
      ? rawConfidence
      : typeof rawConfidence === "string"
      ? Number.parseFloat(rawConfidence.replace("%", ""))
      : 0;
  const confidence = Number.isFinite(parsedConfidence) ? parsedConfidence : 0;
  const rawResult = prediction?.result || "Not available";

  // Extract tumor type label for messaging
  const tumorType = rawResult.toLowerCase().includes("tumor found:")
    ? rawResult.split(":")[1].trim()
    : rawResult;

  useEffect(() => {
    // Only generate AI summary if confidence is >= 60%
    if (!prediction || !grok_api || confidence < 60) {
      setResult(null);
      return;
    }

    const groq = new Groq({
      apiKey: grok_api,
      dangerouslyAllowBrowser: true,
    });

    const getAIResponse = async () => {
      const confidenceText = `${confidence.toFixed(2)}%`;
      const prompt = `
You are a medical assistant. Generate a short, clear HTML response only.

Rules:
- Use only semantic HTML tags such as <h3>, <p>, <ul>, <li>, <strong>.
- No markdown, no code blocks, no triple backticks, no plain text outside HTML.
- If the detected result is "notumor" or "No Tumor", respond with one cheerful congratulatory paragraph like: <p>Congratulations! Based on the prediction, you do not appear to have a tumor. Please continue routine health checkups.</p>
- If the detected result is any type of cancer or tumor, give only medically relevant information in this exact style:
  1. A short heading: <h3>Possible Condition</h3>
  2. A short explanation paragraph
  3. <h3>Symptoms</h3> with 4 bullet points
  4. <h3>Possible Causes</h3> with 3 bullet points
  5. <h3>Effects</h3> with 3 bullet points
  6. <h3>Probable Medical Advice</h3> with 3 bullet points
- Keep the answer concise, practical, and non-alarmist.

Prediction result: ${rawResult}
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
        setResult("<p>Oops! Something went wrong while generating the clinical summary.</p>");
      }
    };

    getAIResponse();
  }, [prediction, grok_api, confidence, rawResult]);

  if (!prediction) {
    return (
      <div className="prediction-container empty-state">
        <h2>Prediction Result</h2>
        <p className="no-prediction">No prediction available yet.</p>
      </div>
    );
  }

  // Evaluate Confidence Brackets & Tailored Messaging
  let confidenceBadgeClass = "low";
  let confidenceMessage = "";
  let showResult = true;

  if (confidence > 85) {
    confidenceBadgeClass = "high";
    confidenceMessage = "High confidence prediction. The model output is most probably correct.";
  } else if (confidence >= 70) {
    confidenceBadgeClass = "medium";
    confidenceMessage = "Moderate confidence prediction. Result is displayed, but it may or may not be correct.";
  } else if (confidence >= 60) {
    confidenceBadgeClass = "warning";
    confidenceMessage = `I think it is ${tumorType}, but I am not sure. You may want to consult a doctor for it.`;
  } else {
    confidenceBadgeClass = "low";
    confidenceMessage = "I am not sure about it. Can you please provide another image or consult with a doctor?";
    showResult = false;
  }

  return (
    <div className="prediction-container">
      <div className="result-header">
        <div>
          <p className="panel-kicker">AI Diagnostic Analysis</p>
          <h2>Prediction Result</h2>
        </div>
        <span className={`confidence-badge ${confidenceBadgeClass}`}>
          {confidence.toFixed(2)}% confidence
        </span>
      </div>

      {/* Confidence Alert & Guidance Banner */}
      <div className={`confidence-alert-banner ${confidenceBadgeClass}`}>
        <p>{confidenceMessage}</p>
      </div>

      {showResult ? (
        <>
          <div className="classification-box">
            <span className="label">Diagnosis</span>
            <strong>{rawResult}</strong>
          </div>

          {result ? (
            <div className="result-html" dangerouslySetInnerHTML={{ __html: result }} />
          ) : (
            <div className="loading-box">
              <div className="spinner" />
              <p>Generating clinical summary...</p>
            </div>
          )}
        </>
      ) : (
        <div className="uncertain-box">
          <h3>Uncertain Diagnostic Confidence</h3>
          <p>{confidenceMessage}</p>
        </div>
      )}
    </div>
  );
};

export default Prediction;
