import Imageinput from "./components/Imageinput";
import Prediction from "./components/Prediction";
import { useState } from "react";
const App = () => {
  const [prediction,setPrediction] = useState(null)


  return (
    <div>
      <Imageinput setPrediction={setPrediction}/>
      <Prediction prediction={prediction}/>
    </div>
  );
}


export default App;