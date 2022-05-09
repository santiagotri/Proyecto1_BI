import "./consultar-css.css";
import { useEffect, useState } from "react";

export default function Resultados(props) {
  const [ textoResultado, setTextoResultado ] = useState("");
  const [ warningResultado, setWarningResultado ] = useState("");
  useEffect(() => {
    if(props.error==="True"){
      setTextoResultado("");
      setWarningResultado(props.warning);
    }else{
      if (props.elegible==="True") {
        setTextoResultado("Elegible");
      } else {
        setTextoResultado("Not Elegible");
      }
      setWarningResultado(props.warning)
    }
    
  },[warningResultado, textoResultado, props.elegible,props.warning,props.error]);

  const handleSubmit = (event)=>{
  }

  return (
    <div>
      <div>
        <div className="bold-p">Result</div>
      </div>

      <div className="resultado">{textoResultado}</div>
      <div className="warningResultado">{warningResultado}</div>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <button type="submit" className="btn-colsanitas">Clear</button>
        </div>
      </form>
    </div>
  );
}
