import Imageinput from "./components/Imageinput";
import Prediction from "./components/Prediction";
import { useState } from "react";

const App = () => {
  const [prediction, setPrediction] = useState(null);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">T</div>
          <div>
            <p className="eyebrow">AI Medical Assistant</p>
            <h1>Brain Tumor Prediction</h1>
          </div>
        </div>
        <div className="status-pill">Clinical screening</div>
      </header>

      <main className="dashboard-grid">
        <section className="panel upload-panel">
          <Imageinput setPrediction={setPrediction} />
        </section>

        <section className="panel result-panel">
          <Prediction prediction={prediction} />
        </section>
      </main>
    </div>
  );
};

export default App;