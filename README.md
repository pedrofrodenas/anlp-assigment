# anlp-assigment
An NLP system that extracts key information from Spanish scholarship announcements (BOE) and generates concise summaries. Combines information extraction techniques with generative language models to transform complex PDF documents into structured data and readable summaries for students and administrators.

--


Document Hierarchy
Top Level: RESOLUCIÓN (Resolution)
This is the main document issued by the Secretary of State for Education.
Second Level: CAPÍTULOS (Chapters)
The document is organized into 7 chapters:

CAPÍTULO I: Objeto y ámbito de aplicación (Object and scope of application)
CAPÍTULO II: Clases y cuantías de las becas (Types and amounts of scholarships)
CAPÍTULO III: Requisitos de carácter general (General requirements)
CAPÍTULO IV: Requisitos de carácter económico (Economic requirements)
CAPÍTULO V: Requisitos de carácter académico (Academic requirements)
CAPÍTULO VI: Verificación y control de las becas (Verification and control of scholarships)
CAPÍTULO VII: Reglas de procedimiento (Procedural rules)

Third Level: ARTÍCULOS (Articles)
Each chapter contains multiple articles (71 in total), which is the smallest level of information unit as you mentioned:
Example structure:

Artículo 1: Objeto y beneficiarios
Artículo 2: Financiación de la convocatoria
Artículo 3: Enseñanzas comprendidas
...continuing through...
Artículo 71: Producción de efectos


--

## Extraction:
Input <- pdf
Output <- dict[dict]
    "Capitulos" -> "Articles" -> (Text, pages)

## Bert to topics/subtopics:
Input <- dict[dict] from extraction
Output <- Dictionary like below (topics -> chapters -> subtopics -> ...)
 + Dump it to json

We decide like 4 main TOPICS:
- RESOLUCION
- REQUIREMENTS
- ...
For each main TOPIC we have a set of subtopics:
- RESOLUCION
    -SUBTOPIC1 [...]

Capitulos -> Decide which topic they belong to [BERT EMBEDDINGS]:
Ex. "Objeto y ámbito de aplicación" - Brief description of the topic: "RESOLUCION: description of becas ..."
Capitulos are separated into the different topics.

Then each article belonging to a certain chapter is further separated into the possible subtopics of that TOPIC using the same procedure.

---

JSON:
- TOPIC1
    - CHAPTERS
        - SUBTOPIC1
            - ARTICLES
                - Text <- Probably a summary of the article using a LLM
                - Pages <- References to the original pages of the article
        - SUBTOPIC2 [...]
- TOPIC2 [...]


From these we can create Hierarchy for an index on the abstract summary.

--

## Abstract Summary:
input <- json
output <- whatever thats pretty

To create the abstract summary:
for each article decide whether it is of general interest or a peculiar one.
Summarize the relevant articles
Put reference to original pages of non relevant articles.

- NOTE: Maybe use the topK in article similarity to decide valuable articles. This can be done for each subtopic, i.e. for each subtopic only put topK articles in the final abstract summary (and put page references for the other articles)




---

Divide into topics with bert:

TOPICS <- We decided them and created for each a small description

T: Topic Text: "TOPICNAME - Brief Description"
Q: Pdf extracted Text: "CHAPTERNAME"

Tokenize(T) -> (N,)
Tokenize(Q) -> (M,)

T_bert: Bert(Tokenize(T)) -> (N, E)
Q_bert: Bert(Tokenize(Q)) -> (M, E)

You compute mean over token dimension
Mean(T_bert) -> (1, E)
Mean(Q_bert) -> (1, E)

(Should be normalized if they are not)
Cosine similarity between the two -> Score.


You do so for each possible topic keeping Q fixed.
You get score for each topic -> argmax is topic assigment for Q

---
