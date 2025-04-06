import { useState } from "react";
import '../App.css'
import axios from "axios";


const Imageinput = ({setPrediction}) => {
    const [image, setImage] = useState(null);

    const handleClick = async (e) => {
        const api_uri = import.meta.env.VITE_REST_API;
        const form = new FormData();
        form.append("image", image);
        e.preventDefault();
        const result = await axios.post('http://localhost:5000/predict', form, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        })
        setPrediction(result.data)
    }


    return (
        <form action="" className="image-input-container">
            <div className="input-field">
                <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files[0])} />
                <button onClick={(e) => handleClick(e)}>predict</button>
            </div>
            <img src={image ? URL.createObjectURL(image) : ""} alt="" className="image-preview" />
        </form>
    );
}

export default Imageinput