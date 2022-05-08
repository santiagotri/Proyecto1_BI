import "./consultar-css.css";
import { useNavigate } from "react-router-dom";

export default function HowToUse() {
  let navigate = useNavigate();

  const irA = (ruta) => {
    navigate(ruta);
  };

  return (
    <div >
      <div>
        <div className="bold-p">How to use?</div>
      </div>

      <div>
        <p>
          The instructions for using our prediction system are in the following
          image. Please use it only if you belong to health personnel qualified
          for this task.
        </p>
      </div>

      <img
        className="img-fluid"
        src="https://live.staticflickr.com/65535/52057692436_4236675cfc_o.png"
        alt="Captura de Pantalla 2022-05-07 a la(s) 5.35.31 p.m."
      ></img>
      <div>
        <div className="bold-p">Results</div>
      </div>

      <div>
        <p>
          When you click at the "Send" button you'll see a result if the pacient
          is or not eligible
        </p>
      </div>

      <form>
        <div className="form-group">
          <button onClick={()=> irA("/")} className="btn-colsanitas">Return</button>
        </div>
      </form>
    </div>
  );
}
