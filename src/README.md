# PySpark Cluster

Este projeto provisiona um cluster Spark simples via Docker Compose (Master + 2 Workers) e inclui um tutorial em PySpark com visualizações e relatório.

## ✔ Setup do cluster

- Pré-requisitos: Docker e Docker Compose instalados.
- Subir o cluster:

```powershell
cd "c:\Users\jp14h\Documents\Humberto-Bianca\pyspark_cluster"; docker compose up -d
```

- Serviços e portas expostas:
  - Master UI: `http://localhost:8080`
  - Workers UI: `http://localhost:8081` e `http://localhost:8082`
  - Spark Application UI (jobs em execução): `http://localhost:4040`
  - History Server: `http://localhost:18080`

O History Server lê os logs de eventos do volume compartilhado `spark-eventlog`, permitindo inspeção de jobs já finalizados.

## ✔ Demonstração do acesso ao Master e History Server

- Acesse o Master UI em `http://localhost:8080` e verifique os workers conectados.
- Rode o tutorial e, após finalizar, abra `http://localhost:18080` para visualizar os jobs concluídos e estágios.

## ✔ Execução bem-sucedida do tutorial

O notebook `app/tutorial.ipynb` demonstra:
- Criação da `SparkSession`.
- Construção de um `DataFrame` simples.
- Transformações (ex.: groupBy, filter) e ações (ex.: show, count).
- Coleta para Pandas e visualização com Matplotlib (contagem por categoria).

Para executar o notebook dentro do cluster, utilize o contêiner do master:

```powershell
cd "c:\Users\jp14h\Documents\Humberto-Bianca\pyspark_cluster"; docker compose exec spark-master bash -lc "pip install -q notebook matplotlib pandas ; jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password=''"
```

Em seguida, acesse `http://localhost:8888` no navegador para abrir o notebook na pasta `/opt/spark/app`.

Durante a execução, faça capturas de tela das visualizações do Matplotlib e das UIs (Master e History Server) e insira-as no relatório conforme necessário.

## ✔ Relatório curto e organizado

O relatório está integrado ao próprio notebook via células Markdown, cobrindo:
- Objetivo do experimento.
- Descrição do dataset sintético e das transformações.
- Interpretação dos resultados (tabelas e gráficos).
- Observações sobre o uso das UIs do Spark e do History Server.

## Encerramento

Para parar e remover o cluster:

```powershell
cd "c:\Users\jp14h\Documents\Humberto-Bianca\pyspark_cluster"; docker compose down
```
