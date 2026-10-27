import pandas as pd

"""
Método para cargar los archivos CSV.
- Retorna un objeto tipo DataFrame con mensajes de texto y sus etiquetas.
"""
def cargar_csv(ruta:str) -> pd.DataFrame:

    TEXT_COLUMN = "mensaje"
    LABEL_COLUMN = "etiqueta_gold"

    try:
        # leemos los archivos
        df = pd.read_csv(ruta,
                        dtype={TEXT_COLUMN: str},
                        keep_default_na=False,)

        
        # convertimos las etiquetas en números
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="raise").astype(int)


        # Retornamos el DataFrame
        return df
    
    except ValueError as ex:
          raise ValueError(f"ocurrio una excepción en carga_CSV : {ex}")