import { useState } from "react";
import "../App.css";
import axios from "axios";

const Imageinput = ({ setPrediction }) => {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleClick = async (e) => {
    e.preventDefault();

    if (!image) {
      return;
    }

    const api_uri = import.meta.env.VITE_REST_API;
    const form = new FormData();
    form.append("image", image);

    try {
      setLoading(true);
      const result = await axios.post(api_uri, form, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setPrediction(result.data);
    } catch (error) {
      console.error("Prediction upload error:", error);
      setPrediction({
        result: "error",
        confidence: 0,
        message: "Unable to process the image. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="image-input-container" onSubmit={handleClick}>
      <div className="upload-copy">
        <p className="panel-kicker">Scan upload</p>
        <h2>Upload MRI or brain image</h2>
        <p className="helper-text">
          Analyze a medical image to detect the likelihood of a tumor.
        </p>
      </div>

      <div className="upload-actions">
        <label className="file-picker" htmlFor="brain-image">
          <input
            id="brain-image"
            type="file"
            accept="image/*"
            onChange={(e) => setImage(e.target.files[0])}
          />
          <span>{image ? image.name : "Choose image"}</span>
        </label>

        <button type="submit" className="predict-btn" disabled={!image || loading}>
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </div>

      <div className="preview-card">
        {image ? (
          <img
            src={URL.createObjectURL(image)}
            alt="Selected diagnostic preview"
            className="image-preview"
          />
        ) : (
          <div className="preview-placeholder">
            <span>No image selected</span>
          </div>
        )}
      </div>
    </form>
  );
};

export default Imageinput;