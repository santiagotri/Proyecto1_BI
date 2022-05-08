import "./consultar-css.css"
import { useNavigate } from "react-router-dom";
import Resultados from "./resultados";
import { useState } from "react";

export default function Consultar(){
  const[resultados, setResultados] = useState("")
  let navigate = useNavigate();

  const handleSubmit = (event)=>{
    event.preventDefault();
    setResultados(<Resultados elegible="False"></Resultados>)
  }

  const irA = (ruta)=>{
    navigate(ruta);
  };

    return (<div>
      {resultados}
            <form onSubmit={handleSubmit}>
            <div className="form-group">
                <label htmlFor="studyInput">Study</label>
                <input type="text" className="form-control" id="studyInput" placeholder="study interventions are Saracatinib"></input>
            </div>
            <div className="form-group">
                <label htmlFor="exampleFormControlTextarea1">Condition </label>
                <textarea className="form-control" id="exampleFormControlTextarea1" rows="5"></textarea>
            </div>
            <div className="form-group">
                <button type="submit" className="btn-colsanitas">Send</button>
            </div>
            
            </form>
            
            <p className="a" onClick={()=>irA("/how-to-use")}>How to use?</p>
    </div>);
  
}
