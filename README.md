# Information_Retrieval_Project
TU Delft IN4325 project

## Task Selection
TREC 2019 Deep Learning Document Ranking Task

## Toolkit
[Pyterrier](https://github.com/terrier-org/pyterrier) 

## Baseline Model
- smoothed Dirichlet Language model
- LamdaMART with features: LM baseline score, TF score, TF-IDF score, DPH score, BM25 score and PL2 score


## Error Analysis 
Distribution of counts with different reciprocal rank values
![image](https://github.com/YaoM1886/Information_Retrieval_Project/blob/main/images/rr_counts.png)

Document length and query length of highest rank, relevant documents
![image](https://github.com/YaoM1886/Information_Retrieval_Project/blob/main/images/doclen.png)

URL depth and query length of highest rank, relevant documents
![image]((https://github.com/YaoM1886/Information_Retrieval_Project/blob/main/images/urldep.png)
)

## Improvement

- pseudo-relevance feedback
- informed document priors

## Results





