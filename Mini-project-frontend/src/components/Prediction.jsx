import React, { useState, useEffect } from "react";
import { GoogleGenAI } from "@google/genai";

const Prediction = ({ prediction }) => {
  const gemini_api = import.meta.env.VITE_GEN_AI;
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!prediction) return;

    const ai = new GoogleGenAI({ apiKey: gemini_api });

    const getAIResponse = async () => {
      const prompt = `
You are a doctor who will give the symptoms and cure for different tumors.
Respond using clean, semantic HTML only — such as <h3>, <ul>, <li>, <p>, <strong>.
Do not use markdown or code blocks. Do not use triple backticks. No extra commentary.
If the user says "notumor", just congratulate them in a cheerful <p> tag.

I have ${prediction.confidence} confidence that I have ${prediction.result}.
`;

      try {
        const response = await ai.models.generateContent({
          model: "gemini-2.0-flash",
          contents: [
            {
              role: "user",
              parts: [{ text: prompt }],
            },
          ],
        });

        const resultText = response.candidates?.[0]?.content?.parts?.[0]?.text;
        setResult(resultText);
      } catch (error) {
        console.error("AI error:", error);
        setResult("<p>Oops! Something went wrong while generating the prediction.</p>");
      }
    };

    getAIResponse();
  }, [prediction]);

  return (
    <div className="prediction-container">
      <h2>Prediction Result</h2>
      {prediction ? (
        result ? (
          <div
            className="result-html"
            dangerouslySetInnerHTML={{ __html: result }}
          />
        ) : (
          <p>Generating response...</p>
        )
      ) : (
        <p className="no-prediction">No prediction available</p>
      )}
    </div>
  );
};

export default Prediction;
