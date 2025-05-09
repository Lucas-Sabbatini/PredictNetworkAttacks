# Como rodar este projeto Python com Jupyter Notebook

## Pré-requisitos

Antes de começar, certifique-se de ter as seguintes ferramentas instaladas no seu sistema:

- **Python 3**:
  ```bash
  sudo apt update
  sudo apt install python3
  ```
- **pip**:
  ```bash
  sudo apt install python3-pip
  ```
- **venv** (módulo para ambientes virtuais, geralmente já incluso no Python 3):
  ```bash
  sudo apt install python3-venv
  ```
- **Jupyter Notebook**: Será instalado automaticamente ao rodar o comando de instalação das dependências, pois está listado no `requirements.txt`.

## 1. Crie um ambiente virtual

Recomenda-se usar um ambiente virtual para isolar as dependências do projeto. Execute os comandos abaixo no terminal, na pasta do projeto:

```bash
python3 -m venv venv
```

- Isso criará uma pasta chamada `venv` com o ambiente virtual.

## 2. Ative o ambiente virtual

```bash
source venv/bin/activate
```

Você verá o nome do ambiente virtual no início da linha do terminal.

## 3. Instale as dependências

Com o ambiente virtual ativado, instale as dependências listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 4. Execute o Jupyter Notebook

Com tudo instalado, rode o Jupyter Notebook:

```bash
jupyter notebook
```

Isso abrirá uma página no seu navegador, caso tenha algum problema utilize o link gerado pelo servidor local Jupyther. Clique no arquivo `CodigoMD.ipynb` para abrir e executar o notebook.

Alternativamente, você pode executar o código procedural .py com o comando abaixo:

```bash
python CodigoProcedural.py
```
