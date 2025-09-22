# Smart Pathfinder 🛰️

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)

**Software de Visão Computacional para Geração Inteligente de Trajetórias**

---

**Smart Pathfinder** é o componente de cliente inteligente do ecossistema **Smart Trajectories**. Desenvolvido em Python, este software processa fontes de vídeo (arquivos ou streams em tempo real) para detectar, rastrear e gerar dados de trajetória otimizados para objetos como veículos e pessoas.

Utilizando algoritmos avançados, o Smart Pathfinder filtra pontos redundantes, sumariza trajetórias longas para extrair a informação mais relevante e envia os dados de forma segura e eficiente para um web service para armazenamento e análise.

## ✨ Principais Funcionalidades

- **Detecção de Alta Performance:** Utiliza o modelo **YOLOv10** para uma detecção de objetos rápida e precisa.
- **Rastreamento Robusto:** Emprega o algoritmo **OC-SORT** para manter a identidade dos objetos mesmo em cenas com oclusões.
- **Coleta de Pontos Inteligente:** Um filtro customizado reduz drasticamente a redundância de dados, registrando novos pontos de trajetória apenas quando um objeto se move uma distância mínima ou muda de direção.
- **Sumarização de Trajetórias:** Para trajetórias muito longas, um algoritmo de seleção inteligente (`select_best_points`) reduz a trajetória a um número fixo de pontos (ex: 50), mantendo os vértices mais importantes que definem sua forma.
- **Envio Automatizado em Lotes:** As trajetórias finalizadas são enviadas em lotes para um backend, usando threads para não bloquear o processamento de vídeo.
- **Comunicação Segura:** As requisições para o web service são autenticadas usando uma chave de API (`X-API-KEY`).
- **Altamente Configurável:** Todos os parâmetros essenciais (fonte de vídeo, ROI, endpoint, configurações do tracker, etc.) são gerenciados através de um único arquivo `data.yaml`.

---

## 🚀 Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Visão Computacional:** OpenCV, Ultralytics (YOLOv10)
- **Rastreamento:** OC-SORT
- **Cálculos Numéricos:** NumPy
- **Comunicação:** Requests
- **Configuração:** PyYAML
- **Dependências Adicionais:** FilterPy

---

## ⚙️ Instalação e Uso

Siga os passos abaixo para configurar e executar o Smart Pathfinder.

### Pré-requisitos

* Python 3.10 ou superior
* Git
* (Opcional, para melhor performance) Uma GPU NVIDIA com drivers e CUDA configurados.

### 1. Clone o Repositório

```bash
git clone [https://github.com/RodrigoMaia03/Smart-Pathfinder](https://github.com/RodrigoMaia03/Smart-Pathfinder)
cd Smart-Pathfinder
```

### 2. Crie e Ative um Ambiente Virtual

```bash
# Crie o ambiente
python -m venv venv

# Ative o ambiente
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
# source venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Configure o data.yaml

Crie um arquivo chamado data.yaml na raiz do projeto. Use o exemplo abaixo como modelo e preencha com as suas configurações.

```bash
video_stream_link: null
arquivo: arquivo.mp4
camera_name: "Nome_camera"
show_stream:
- true
send_image:
- true
classes:
- - 0
  - 1
  - 2
  - 3
  - 5
  - 6
  - 7
id_camera:
- 1
contador_tempo:
- false
limite_tempo:
- 5
device:
- 0
socket:
- true
fps:
- 24
endpoint: endpoint-smart-trajectories-aqui
api_key: "sua-chave-forte-api"
data:
- 27-04-2023 08:16:12
lines:
- - id: 1
    start:
    - 453
    - 217
    end:
    - 775
    - 214
  - id: 2
    start:
    - 400
    - 643
    end:
    - 1357
    - 634
roi_points:
- - 237
  - 74
  - 1046
  - 637
polygons:
- []
```

### 5. Execute a Aplicação

Com o ambiente virtual ativado e o data.yaml configurado, inicie o script principal (vamos assumir que se chama poc.py):

```bash
python poc.py
```

---

## 🤝 Agradecimentos

* **Autor:** Rodrigo Maia
* **Orientador:** Prof. Paulo Rego

---
