# Descrição das Colunas do CSV

## `Image Index`

O nome exato do ficheiro de imagem (ex: `00000001_000.png`).

## `Finding Labels` (A mais importante)

O diagnóstico médico. Nota que pode ser "No Finding" (Saudável) ou ter várias doenças separadas por `|` (ex: `Cardiomegaly|Effusion`).

## `Follow-up #`

O número sequencial do exame para aquele paciente.

* `0`: Primeira vez que o paciente fez o Raio-X.
* `1`: Segundo Raio-X (acompanhamento), etc.
* **Importância:** Para este projeto específico, **não é crítica**. Mas num projeto real, serve para ver a evolução da doença (ex: "A pneumonia diminuiu entre o exame 0 e o 2?").

## `Patient ID` (CRÍTICA PARA O TREINO)

O número de identificação único do paciente.

* Usar *esta* coluna para dividir o teu dataset.
* *Regra de Ouro:* O Paciente `1` nunca pode estar no Treino e no Teste ao mesmo tempo. Se estiver, o modelo decora a anatomia do Paciente 1 em vez de aprender o que é Cardiomegalia.

## `Patient Age`

Idade em anos (na maioria dos casos).

## `Patient Gender`

`M` (Masculino) ou `F` (Feminino).

## `View Position` (PA vs AP)

A posição em que o Raio-X foi tirado.

* **PA (Posterior-Anterior):** O Raio-X entra pelas costas e sai pelo peito. É o padrão (paciente em pé).
* **AP (Anterior-Posterior):** O Raio-X entra pelo peito. Comum em pacientes acamados (UTI/Emergência).

No AP, o coração parece **maior** do que realmente é (porque está mais longe do filme/sensor). O modelo pode achar que é *Cardiomegalia* erradamente.

## `OriginalImage[Width Height]`

A resolução original da imagem (ex: 2682 x 2749 pixéis).

## `OriginalImagePixelSpacing[x y]`

O tamanho físico de cada pixel em milímetros (ex: 0.143 mm).
