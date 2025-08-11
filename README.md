# AIM 2

# Installation instructions
```bash 
pip install -e .
pip install -r requirements.txt # use appropriate OS.

# for models used for sentence splitting (can be use for other purposes)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz # for non-transformer model 
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz # for transformer model which requires pytorch cuda

```
# Installation for Mac (untested)
```bash
pip install -e .

pip install scispacy
# for models used for sentence splitting (can be use for other purposes)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz # for non-transformer model 
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz # for transformer model which requires pytorch cuda
pip install dotenv
pip install schemic

pip install bitsandbytes # don't know if this works for macOS
pip install accelerate  # dont know if this works for macOS.

pip install vllm # future use - has a lot of benefits for local inferencing - relationship step much later.
pip install outlines # ignore the warning!
```
# API tokens
use the `.env.example` to create a `.env` file to hold your API keys.

## vs. GPT-NER

| Dimension                   | **Our BioC pipeline (v1)**                                                                                                     | **GPT-NER (paper/repo)**                                                                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input format**            | BioC XML passages → paragraph windows (Intro/Results/Discussion, etc.). Keeps stand-off offsets + section provenance.          | Expects **MRC-style JSON** (each label is a “query” over the sentence). You’d add a BioC→MRC export step to use their code. [[BioC docs][1]] [[MRC-NER repo][2]]                                      |
| **Prompting granularity**   | Per paragraph. Either one prompt with all 9 labels or (better) per-label prompts.                                               | **Per-label prompts** by design (iterate over labels due to prompt length constraints). Output wraps entities with `@@ … ##`.                                                                            |
| **Few-shot retrieval**      | Static: label descriptions + a few hard-coded domain examples. (Optional) add **sentence-level** retrieval (SimCSE/E5).   | kNN retrieval is core: **sentence-level** (SimCSE) and stronger **entity-level** using a **fine-tuned NER tagger** to get token embeddings. Big gains come from entity-level. [[SimCSE][3]][[paper][4]] |
| **Self-verification**       | Add a cheap, local LLM **yes/no verifier** (e.g., Outlines + a small model) with typed outputs (`Literal["yes","no"]`. Can be done using GPT as well.) OR programmatically against files with all possible values.             | Built-in **self-verification** pass (LLM answers yes/no) with kNN-retrieved demos; reduces over-prediction/hallucination. [[Outlines][5]]                                                                |
| **Label schema**            | Totally custom (our 9 biomedical labels). We own the schema and JSON.                                                           | Also flexible—labels are text—but repo/eval assume MRC datasets; extra plumbing for custom schemas. [[MRC-NER repo][2]]                                                                                 |
| **Offsets & auditability**  | Natural fit: BioC keeps doc-level offsets and sections; easy provenance/audits.                                                 | Doable, but the repo focuses on span eval; you’ll manage your own offsets if converting back to BioC. [[BioC docs][1]]                                                                                  |
| **Nested entities**         | Emit separate spans; BioC style makes this straightforward.                                                                     | Evaluated on nested datasets (ACE/GENIA) via per-label prompting + span parsing.                                                                                                                         |
| **Data need to get strong** | Good out of the gate. Later, add sentence-level retrieval; **optionally** train a tiny tagger to unlock entity-level kNN.       | Needs a **fine-tuned NER tagger** to reach best numbers (for token-level kNN). Without it, sentence-level is weaker.                                                                                     |
| **Engineering lift**        | Light: BioC parser, prompts, JSON schema, verifier, and a simple merge/dedup.                                                   | Heavier if you adopt the whole stack: MRC preprocessing, retrieval indices, per-label runs, verification, eval scripts. [[MRC-NER repo][2]]                                                             |
| **Cost profile**            | You control it: restrict to key sections, per-label calls only if needed, local verifier is cheap.                              | Cost scales with **(#sentences × #labels)** plus verification calls; demos add tokens too. Original paper used GPT-3 (davinci-003).                                                                      |
| **When it shines**          | Niche biomedical labels; strict JSON; low budget; BioC-native outputs + easy audits.                                            | When you can add **entity-level kNN** and lots of solid demos; very strong in few/low-resource settings.                                                                                                 |

[1]: https://bioc.readthedocs.io/
[2]: https://github.com/ShannonAI/mrc-for-flat-nested-ner
[3]: https://github.com/princeton-nlp/SimCSE
[4]: https://arxiv.org/abs/2104.08821
[5]: https://github.com/dottxt-ai/outlines
