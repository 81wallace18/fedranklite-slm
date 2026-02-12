## Paper base (FAH-QLoRA)

### Análise

Propõe um framework de fine-tuning federado que combina **quantização do backbone** + **LoRA com ranks adaptativos e heterogêneos** para reduzir **memória** e **tempo wall-clock**, com **análise de convergência** e experimentos em tarefas do GLUE.  

### Problema

Fine-tuning federado de modelos grandes é caro em **memória** e **tempo**, e a heterogeneidade dos clientes (compute/rede) causa **stragglers**, atrasando rodadas.  

### Motivação

Fazer fine-tuning com **privacidade** (dados locais) viável em edge/heterogêneo, já que mesmo com LoRA o backbone ainda pesa e o custo/latência inviabiliza na prática.  

### Objetivos

* Reduzir **memória** no cliente (quantização do backbone). 
* Reduzir **tempo wall-clock** (rank LoRA adaptativo e heterogêneo). 
* Mitigar **stragglers**. 
* Dar **garantia teórica** (convergência). 

### Relevância

Ataca um gargalo real de adoção de FL+LLM: tornar o ajuste federado **mais barato e rápido** sob heterogeneidade e restrições de recurso. 

### Aplicação

Cenários em que **não dá pra centralizar dados** e você precisa adaptar um modelo com clientes heterogêneos; avaliado em tarefas de NLP padronizadas (GLUE).  

### Cenário

Servidor + N clientes, cada um com dataset local e perdas locais; modelo base pré-treinado + LoRA como módulo treinável; objetivo global é minimizar a perda média federada. 

### Metodologia

* **Quantizar/dequantizar** o backbone para reduzir footprint. 
* Loop federado: servidor envia LoRA, clientes treinam, devolvem updates, servidor agrega. 
* Escolha de ranks em **dois estágios**: (i) rank médio por rodada maximizando “loss decrease rate”; (ii) ranks por cliente resolvendo o problema P1 para minimizar tempo de rodada (com relaxação + CVX + arredondamento).  
* Agregação com ranks diferentes via **truncation + zero-padding**. 

### Métricas

* Métrica operacional do método: **loss decrease rate** por tempo. 
* Experimentos: **tempo wall-clock**, **memória pico**, e métricas de tarefa (accuracy / Pearson).  

### Contribuição

* Framework FAH-QLoRA (quantização + LoRA rank adaptativo/heterogêneo). 
* Estratégia de seleção de ranks em 2 estágios (inclui P1). 
* Agregação de LoRA heterogêneo (padding/truncation) + prova de convergência + resultados experimentais. 

---

## O que **nós** vamos fazer de diferente

A diferença não é “mudar tudo”, e sim **trocar o coração** do método e **mudar o cenário-alvo** para ficar SLM-first e mais “deployable”.

### Problema (nosso recorte)

Mesmo objetivo geral (FL+PEFT eficiente), mas agora com foco em **SLMs** e execução realista: heterogeneidade forte, não-IID severo e rodadas com **deadline**.

### Motivação (nosso porquê)

* “Quero algo que eu consiga rodar e escalar de verdade” com seus tiers (GPU 24GB/32GB e CPU pesado no Apolo).
* Um scheduler que depende de CVX/otimização pesada pode ser um gargalo prático quando você aumenta número de clientes/rodadas.

### Objetivos (nossos)

1. **Eliminar CVX/solver**: criar um **scheduler leve/online** para decidir rank (e opcionalmente bits) por cliente/rodada.
2. Otimizar **time-to-target** (tempo até bater um score), sob **deadlines** e stragglers.
3. Validar em **SLM-first** e **não-IID severo**, com métricas de fairness e estabilidade.

### Relevância (nossa)

A contribuição vira “systems + ML”: tornar FL+SLM **prático e escalável**, com decisões rápidas e robustas — não só bom no papel.

### Aplicação (nossa)

Fine-tuning federado de SLMs em ambientes com:

* mistura GPU/CPU,
* restrição de rede,
* domínios diferentes por cliente (não-IID),
* exigência de previsibilidade (deadlines).

### Cenário (nosso)

Vamos explicitar tiers (isso é diferencial experimental):

* **Clientes rápidos (GPU 24GB)** nos seus servers,
* **Clientes médios (GPU 32GB)** no node do Apolo,
* **Clientes lentos (CPU)** no Apolo 256 cores, com limitação de threads para simular “edge/straggler”.

### Metodologia (nossa)

* Mantemos: **backbone quantizado + LoRA** (mesma base conceitual do paper).
* Mudamos o principal: **rank allocation sem CVX**:

  * regra/heurística online (greedy/bandit) baseada em sinais observáveis (tempo da rodada anterior, throughput, queda de loss local, deadline).
* (Opcional, se quiser mais forte): também escolher **bits de quantização por cliente** (co-adaptação rank+bits).

### Métricas (nossas)

Além de accuracy/score:

* **time-to-target** (principal),
* % de rodadas que respeitam **deadline**,
* bytes transmitidos/rodada (se você incluir comunicação),
* fairness (pior cliente / p10),
* memória pico por tier.

### Contribuição (nossa)

1. Um **scheduler leve e deployable** (sem CVX) para rank (e opcional bits) em FL+SLM.
2. Um **protocolo experimental realista** (tiers GPU/CPU + rede + deadlines + não-IID severo).
3. Evidência de ganhos em **time-to-target** e robustez a stragglers, com ablações claras.
