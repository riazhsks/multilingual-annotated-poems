# This repository contains 

:one: A collection of seven poetry corpora in five languages (English, German, Spanish, Czech, Hungarian) enriched with annotations.

:two: Trained models for automatic classification of poetic forms:

- Four Logistic Regression models trained on originally labeled CCV data

- Four Logistic Regression models trained on the English corpus

- Four Support Vector Classifier models trained on the English corpus

- Eight feature set weights configuration for rule-based classification

:three: Source codes used for experiments with automatic classification of poetic forms:

- Data processing
- Rule-based classification
- Supervised machine learning (Logistic Regression and Support Vector Classifier)
- Unsupervised machine learning (*k*-means ans HDBSCAN)
- Prompting Large Language Models (```gemini-2.5-pro``` and ```gpt-4o```)

## :one: Annotated poetry collection

### Full annotations

Annotation format
```
{
  "title": the poem title (string), 
  "normalized_title": normalized* version of title, 
  "author": the poet’s name and surname separated by a single space (or in the source form) (string),
  "collection": "the title of the collection containing the poem (string),
  "normalized_collection": normalized* version of collection, 
  "form": formal element, fixed, or unfixed poetic form type (e.g. sonnet, couplet, ode) (string), 
  "stanza_scheme": the numeric sequence representing the number of lines in each stanza (string, e.g., “4 4 3 3”), 
  "rhyme_scheme": the numeric sequence representing the rhyming pattern (string, e.g., “1 2 3 2 1 2 1 2 4 5 3 6 7 3”),
  "rhyme_scheme_per_stanza": the rhyme scheme segmented by the stanza structure (string, e.g., “1232 1212 453 673”),
  "metrical_foot": the metrical foot of the poem (string, e.g., “iambic”, or “polymetric”), 
  "metrical_foot_count": the number of metrical feet in the poem (string, e.g., “3”, or “mixed”), 
  "line_count": the number of lines in the poem (integer), 
  "total_syllables": the number of syllables in the poem (integer), 
  "average_syllable_count": the mean number of syllables per line (float), 
  "text": the text of the poem (string),
  "normalized_text": normalized* version of text
}
*normalization process includes tokenization, stopword removal, lemmatization, lowercasing and non-alphanumeric characters removal 

If any feature could not be extracted, it is represented as null for strings and 0 for numbers.
```
---
Corpus of Czech verse (CCV)

Link: https://github.com/versotym/corpusCzechVerse

```
@article{ccv2015,
	author = {Plecháč, Petr and Kolár, Robert},
	title = {The Corpus of Czech Verse},
	doi = {10.12697/smp.2015.2.1.05},
	journal = {Studia Metrica et Poetica},
	number = {1},
	volume = {2},
	year = {2015},
	pages = {107--118},
}

@article{ccv2016,
	author = {Plecháč, Petr},
	title = {Czech Verse Processing System KVĚTA -- Phonetic and Metrical Components},
	doi = {10.1515/glot-2016-0013},
	journal = {Glottotheory},
	number = {2},
	volume = {7},
	year = {2016},
	pages = {159--174},
}
```
---
English corpus

Link: https://github.com/maria-antoniak/poetry-eval
```
@inproceedings{walsh-etal-2024-sonnet,
    title = "Sonnet or Not, Bot? Poetry Evaluation for Large Models and Datasets",
    author = "Walsh, Melanie  and
      Preus, Anna  and
      Antoniak, Maria",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = "nov",
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.914/",
    doi = "10.18653/v1/2024.findings-emnlp.914",
    pages = "15568--15603"
}
```
---
The Diachronic Spanish Sonnet Corpus/CorpusSonetosSigloDeOro 

Link: https://github.com/bncolorado/CorpusSonetosSigloDeOro

```
@article{10.1093/llc/fqaa035,
    author = {Ruiz Fabo, Pablo and Bermúdez Sabel, Helena and Martínez Cantón, Clara and González-Blanco, Elena},
    title = {The Diachronic Spanish Sonnet Corpus: TEI and linked open data encoding, data distribution, and metrical findings},
    journal = {Digital Scholarship in the Humanities},
    volume = {36},
    number = {Supplement\_1},
    pages = {i68-i80},
    year = {2020},
    month = {09},
    abstract = {How has the sonnet form in Spanish evolved over the centuries? What is the distribution of metrical patterns and combinations thereof, considering diachronic, geographical, and social factors? What rhyme schemes are favoured in different periods and regions? How is enjambment distributed within the sonnet? Providing quantitative answers to such questions requires a corpus spanning several centuries, annotated for the relevant literary features and containing author metadata. The absence of appropriate digital resources to undertake a macroanalytic study of the evolution of the sonnet in Spanish led us to create the Diachronic Spanish Sonnet Corpus. This article presents how the corpus was designed for providing quantitative evidence on the evolution of sonnets in Spanish, and our findings regarding metrics and enjambment. The corpus contains 4,085 sonnets by 1,204 Spanish and Latin American authors (15th to 19th centuries), encoded in TEI, with RDFa attributes. The corpus aims at breadth, including many peripheral authors besides some major ones. Author metadata were encoded (dates, origin, gender). Scansion and enjambment were annotated automatically, with the ADSO and ANJA tools. The range of authors and periods, the use of TEI and RDFa for interoperability, and the combination of metrical and enjambment annotations goes beyond previously available digital resources. The corpus allowed us to examine the evolution of metrical patterns and their combinations after the Golden Age, complementing earlier studies. We also observed an increase in enjambment across the tercets in the 19th century, which may indicate increased variety in the discourse organization of sonnets in the period.},
    issn = {2055-7671},
    doi = {10.1093/llc/fqaa035},
    url = {https://doi.org/10.1093/llc/fqaa035},
    eprint = {https://academic.oup.com/dsh/article-pdf/36/Supplement_1/i68/40506370/fqaa035.pdf},
}
```
---
Diachronic Spanish Sonnet Corpus (DISCO)

Link: https://github.com/pruizf/disco
```

@misc{ruizfabo2017disco,
  title={{Diachronic Spanish Sonnet Corpus (DISCO)}},
  author={Ruiz Fabo, Pablo and Berm{\'u}dez Sabel, Helena and Mart{\'i}nez Cant{\'o}n, Clara and Calvo Tello, Jos{\'e}},
  year={2017},
  publisher={UNED},
  address={Madrid},
  url={https://github.com/pruizfabo/disco}
}
```
---
A German Poetry Corpus / Deutsches Lyrik Korpus (DLK)

Link: https://github.com/tnhaider/DLK

```
@article{haider2021metrical,
  title={Metrical Tagging in the Wild: Building and Annotating Poetry Corpora with Rhythmic Features},
  author={Haider, Thomas},
  journal={Proceedings of the European Association for Computational Linguistics, arXiv:2102.08858},
  year={2021}
}

@inproceedings{haider2019semantic,
  title={Semantic Change and Emerging Tropes In a Large Corpus of New High German Poetry},
  author={Haider, Thomas and Eger, Steffen},
  booktitle={Proceedings of the 1st International Workshop on Computational Approaches to Historical Language Change},
  pages={216--222},
  year={2019},
  source={https://www.aclweb.org/anthology/W19-4727}
}
```
---

### Standalone annotations

Annotation format
```
{
  "title": the poem title (string), 
  "normalized_title": normalized* version of title, 
  "author": the poet’s name and surname separated by a single space (or in the source form) (string),
  "collection": "the title of the collection containing the poem (string),
  "normalized_collection": normalized* version of collection, 
  "form": formal element, fixed, or unfixed poetic form type (e.g. sonnet, couplet, ode) (string), 
  "stanza_scheme": the numeric sequence representing the number of lines in each stanza (string, e.g., “4 4 3 3”), 
  "rhyme_scheme": the numeric sequence representing the rhyming pattern (string, e.g., “1 2 3 2 1 2 1 2 4 5 3 6 7 3”),
  "rhyme_scheme_per_stanza": the rhyme scheme segmented by the stanza structure (string, e.g., “1232 1212 453 673”),
  "metrical_foot": the metrical foot of the poem (string, e.g., “iambic”, or “polymetric”), 
  "metrical_foot_count": the number of metrical feet in the poem (string, e.g., “3”, or “mixed”), 
  "line_count": the number of lines in the poem (integer), 
  "total_syllables": the number of syllables in the poem (integer), 
  "average_syllable_count": the mean number of syllables per line (float)
}
*normalization process includes tokenization, stopword removal, lemmatization, lowercasing and non-alphanumeric characters removal 

If any feature could not be extracted, it is represented as null for strings and 0 for numbers.
```
---
The Corpus of Contemporary Czech Poetry (C3P)
```
@article{10.1093/llc/fqac013,
    author = {Škrabal, Michal and Piorecký, Karel},
    title = {The Corpus of Contemporary Czech Poetry: A database for research on contemporary poetic language across media},
    journal = {Digital Scholarship in the Humanities},
    volume = {37},
    number = {4},
    pages = {1240-1253},
    year = {2022},
    month = {04},
    abstract = {Our article reports on the emerging Corpus of Contemporary Czech Poetry and the possibilities for its use. We describe the genesis of the idea of creating a specific corpus that combines the principles of synchronicity and genre instead of relying on the presence of poetry in the general corpus of contemporary Czech. We also characterize the structure of our corpus, which is designed to cover both of the basic media areas in which contemporary poetry is published and distributed: either in books or through open publishing platforms on the Internet (literary forums). We additionally describe the functionalities of the tools for mining the corpus data, which are designed to easily serve comparative analyses across media (print/web). We suggest how useful quantitative data analysis can be in the first phase of language-oriented literary research; or rather we point out the necessity of combining quantitative and qualitative approaches. Only the researcher’s interpretative proficiency can decide on the boundaries of the field under study and the meaning of the elements present in it. In text-centred analyses, language corpora should start to play a similar role as other tools of scientific infrastructure, such as bibliographic databases.},
    issn = {2055-7671},
    doi = {10.1093/llc/fqac013},
    url = {https://doi.org/10.1093/llc/fqac013},
    eprint = {https://academic.oup.com/dsh/article-pdf/37/4/1240/46607898/fqac013.pdf},
}
```
---
ELTE Poetry Corpus

Link: https://github.com/ELTE-DH/poetry-corpus
```
@inproceedings{horvath2022elte,
  title = {{ELTE Poetry Corpus}: A Machine Annotated Database of Canonical Hungarian Poetry},
  author = {Horv{\'a}th, P{\'e}ter and Kundr{\'a}th, P{\'e}ter and Indig, Bal{\'a}zs and Fellegi, Zs{\'o}fia and Szl{\'a}vich, Eszter and Bajz{\'a}t, T{\'i}mea Borb{\'a}la and S{\'a}rk{\"o}zi-Lindner, Zs{\'o}fia and Vida, Bence and Karabulut, Aslihan and Tim{\'a}ri, M{\'a}ria and Palk{\'o}, G{\'a}bor},
  booktitle = {Proceedings of the 13th Conference on Language Resources and Evaluation (LREC 2022)},
  pages = {3471--3478},
  year = {2022},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  language = {english}
}
```

## :two: Models
- cz* models are Logistics Regression models trained on originally labeled part of CCV
- eng-lr* models are Logistics Regression models trained on the English corpus
- eng-svc* models are Support Vector Classifier models trained on the English corpus
- {cz, eng-lr, eng-svc}-f-d are models with downsampled *sonnets* (English) or *sonets* (Czech)
- {cz, eng-lr, eng-svc}-f-s are models without downsampling
- {cz, eng-lr, eng-svc}-d-d are delexicalized models with downsampled *sonnets* (English) or *sonets* (Czech)
- {cz, eng-lr, eng-svc}-d-s are delexicalized models without downsampling


## :three: Source code
The source code in this corpus has been developed as a part of my Bachelor thesis.

It includes scripts for:
- Data processing: unificating and annotating corpora.
- Rule-based classification approach: getting rules from statistics, finding optimal weights and predicting poetic forms.
- Supervised learning approach: training LR and SVC models (same as :two:) and predicting poetic forms.
- Delexicalized transfer: predicting poetic forms using delexicalized models obtained with SL approach.
- Unsupervised learning approach: *k*-means and HDBSCAN for clustering by poetic forms.
- Prompting Large Language Models: predicting poetic forms using prompting LLMs.
- Evaluation of performance: combining predictions from all approaches (except for clustering) and evaluating correctness.
