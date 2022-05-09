import "./consultar-css.css"
import { useNavigate } from "react-router-dom";
import Resultados from "./resultados";
import { useState } from "react";

const urlBack= "http://casatrianaserver.local:50000"

export default function Consultar(){
  const[resultados, setResultados] = useState("")
  const [inputs, setInputs] = useState("");
  let navigate = useNavigate();


  const handleChange = (event) => {
    setInputs({ ...inputs, [event.target.name]: event.target.value });
  };

  const handleSubmit = (event)=>{
    event.preventDefault();
    fetch(urlBack,{
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          study: inputs.study,
          condition: inputs.condition,
        }),
      })
        .then((res) => res.json())
        .then((res) => {
          if (res.success) {
            const token = res.token;
          } else {
            const errorMsg = res.message;
          }
        })
        .catch((error) => {
          console.error(error);
          setResultados(<Resultados error="True" elegible="False" warning={"Se ha detectado un error:"+error.toString()} ></Resultados>)
        });
    setResultados(<Resultados error="False" elegible="False" warning="Se ha detectado un error con la(s) palabra(s) [Gato, feliz]"></Resultados>)
  }


  const irA = (ruta)=>{
    navigate(ruta);
  };

    return (<div>
      {resultados}
            <form onSubmit={handleSubmit}>
            <div className="form-group">
                <label htmlFor="studyInput">Study</label>
                <input onChange={handleChange} name="study" type="text" className="form-control" id="studyInput" placeholder="study interventions are Saracatinib"></input>
            </div>
            <div className="form-group">
                <label htmlFor="exampleFormControlTextarea1">Condition </label>
                <textarea onChange={handleChange} name="condition" className="form-control" id="exampleFormControlTextarea1" rows="5"></textarea>
            </div>
            <div className="form-group">
                <button type="submit" className="btn-colsanitas">Send</button>
            </div>
            
            </form>
            
            <p className="a" onClick={()=>irA("/how-to-use")}>How to use?</p>
    </div>);
  
}
