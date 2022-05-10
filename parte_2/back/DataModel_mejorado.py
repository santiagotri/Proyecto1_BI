from multiprocessing.dummy import Array
from pydantic import BaseModel

class DataModelMejorado(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    adult_mortality: list
    infant_deaths: list
    alcohol: list
    percentage_expenditure: list
    hepatitis_B: list
    measles: list
    bmi: list
    under_five_deaths: list
    polio: list
    total_expenditure: list
    diphtheria: list
    hiv_aids: list
    gdp: list
    population: list
    thinness_10_19_years: list
    thinness_5_9_years: list
    income_composition_of_resources	: list
    schooling: list
    life_expectancy:list

#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ["Adult Mortality", "infant deaths", "Alcohol","percentage expenditure","Hepatitis B", "Measles", "BMI",
                "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "DGP", "Population",
                "thinness 10-19 years", "thinness 5-9 years", "Income composition of resources", "Schooling","Life expectancy"]
        