import "./consultar-css.css";
import { useEffect, useState } from "react";

export default function Resultados(props) {
  const [ textoResultado, setTextoResultado ] = useState("");
  useEffect(() => {
    if (props.elegible==="True") {
      setTextoResultado("Elegible");
    } else {
      setTextoResultado("Not Elegible");
    }
  },[textoResultado, props.elegible]);

  const handleSubmit = (event)=>{
  }

  return (
    <div>
      <div>
        <div class="bold-p">Result</div>
      </div>

      <div class="resultado">{textoResultado}</div>
      <form onSubmit={handleSubmit}>
        <div class="form-group">
          <button type="submit" class="btn-colsanitas">Clear</button>
        </div>
      </form>
    </div>
  );
}
