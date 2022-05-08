import ReactDOM from "react-dom";
import Consultar from "./components/consultar";
import HowToUse from "./components/howToUse";
import React, { Component } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import "./components/consultar-css.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";


class Main extends Component {
  
  render() {
    return (
      <BrowserRouter>
        <div className="contenedor-central">
          <img
            className="logo-img"
            src="https://www.newhealthfoundation.org/wp-content/uploads/2018/10/Logo-Colsanitas.png"
            alt="Logo colsanitas"
          ></img>

          <div className="row">
            <Routes>
              <Route path="/" element={<Consultar />} />
              <Route path="how-to-use" element={<HowToUse />} />
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    );
  }
}

ReactDOM.render(<Main />, document.getElementById("root"));
