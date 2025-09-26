# NEL Inference Tool

A command-line tool for **Named Entity Linking (NEL)** that finds the best matches for entity mentions in a gazetteer using semantic similarity.

---

## What This Tool Does

This script takes a list of entity mentions (like `"heart attack"` or `"aspirin"`) and finds the most similar terms in a reference gazetteer (a dictionary of standardized terms with codes).  

It uses **sentence transformers** to convert text into vectors and computes **cosine similarity** to rank matches.

**Workflow**:  
Input: Entity mentions → Process: Vector similarity matching → Output: Top-k gazetteer matches with scores


---

## Quick Start

```bash
pip install -r requirements.txt

python nel_inference.py --gazetteer gazetteer.tsv --input input.tsv --output results.tsv
```

---

## File Formats

### Gazetteer

A gazetteer is a dictionary that links terms to their corresponding codes in a terminology or ontology. The gazetteer must be a TSV format file containing two columns: `term` and `code`. Additional columns are allowed but will not be used by the tool.

#### Gazetteer Example (SNOMED CT terms) in Spanish:

```tsv
term	code
Infarto de miocardio	22298006
Infarto agudo de miocardio	57054005
Dolor de cabeza	25064002
Ácido acetilsalicílico	412586006
Migraña	37796009
Hipertensión arterial	38341003
Diabetes mellitus	73211009
Asma	195967001
Fiebre	386661006
Tos	49727002
Dolor abdominal	21522001
```

### Input

#### Input Example in Spanish:

The input must be a TSV format file containing a column named span with the NER mentions to be linked. Additional columns are also allowed but will not be used by the tool.

```tsv
span
Ataque al corazón
dolor fuerte d cabeza
aspirina
Migraña
hipertension
DM
```

### Output

#### Output Example in Spanish:

The output will maintain the exact same format as the input file, with three additional columns:

* `codes`: list of predicted codes for each mention

* `terms`: list of corresponding terms for each code

* `similarities`: similarity scores for each mention-code/term linking

```tsv
span	codes	terms	similarities
Ataque al corazón	['22298006', '57054005']	['Infarto de miocardio', 'Infarto agudo de miocardio']	[0.7058, 0.6472]
dolor fuerte d cabeza	['25064002', '37796009']	['Dolor de cabeza', 'Migraña']	[0.8958, 0.6891]
aspirina	['412586006', '25064002']	['Ácido acetilsalicílico', 'Dolor de cabeza']	[0.8445, 0.4353]
Migraña	['37796009', '25064002']	['Migraña', 'Dolor de cabeza']	[1.0, 0.7927]
hipertension	['38341003', '73211009']	['Hipertensión arterial', 'Diabetes mellitus']	[0.8953, 0.5399]
DM	['73211009', '38341003']	['Diabetes mellitus', 'Hipertensión arterial']	[0.8794, 0.5147]
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-g, --gazetteer` | Path to gazetteer TSV file | `gazetteer.tsv` |
| `-i, --input` | Path to input TSV file | `input.tsv` |
| `-o, --output` | Path for output TSV file | `output.tsv` |
| `-m, --model` | Sentence transformer model name/path | `ICB-UMA/ClinLinker-KB-GP` |
| `-k, --top_k` | Number of top candidates to retrieve | `10` |
| `-s, --store_vector_db` | Path to save computed vector database | `None` |
| `-v, --vector_db_file` | Path to load pre-computed vector database | `vector_db.pt` |
