import numpy as np
import pandas as pd
import pyterrier as pt
import xgboost as xgb
import time
if not pt.started():
  pt.init(mem=20000)
dataset = pt.get_dataset("trec-deep-learning-docs")
def msmarco_generate():
    with pt.io.autoopen('msmarco-docs.tsv.gz', 'rt') as corpusfile:
        for l in corpusfile:
            docid, url, title, body = l.split("\t")
            yield {'docno' : docid, 'url' : url, 'title' : title, 'text' : body}
props = {
  'indexer.meta.reverse.keys':'docno',
  'termpipelines' : '',
}
indexer = pt.IterDictIndexer("./document_index", blocks=True, verbose=True)
indexer.setProperties(**props)
indexref = indexer.index(msmarco_generate(), fields=['docno', 'text'], meta=['docno', 'text'], meta_lengths=[20, 4096])
index = pt.IndexFactory.of(indexref)
print(index.getCollectionStatistics().toString())

fbr = pt.FeaturesBatchRetrieve(index,
                               properties={"termpipelines": ""},
                               controls = {"wmodel": "DirichletLM"},
                               verbose=True,
                               features=["WMODEL:Tf", "WMODEL:PL2", "WMODEL:BM25", "WMODEL:DPH", "WMODEL:TF_IDF", "SAMPLE"]) % 100
params = {'objective': 'rank:ndcg',
          'learning_rate': 0.1,
          'gamma': 1.0, 'min_child_weight': 0.1,
          'max_depth': 6,
          'verbose': 2,
          'random_state': 42
         }
BaseLTR_LM = fbr >> pt.pipelines.XGBoostLTR_pipeline(xgb.sklearn.XGBRanker(**params))

train_start_time = time.time()
BaseLTR_LM.fit(pt.io.read_topics("sample_train_20000.txt", format="singleline"), dataset.get_qrels("train"), dataset.get_topics("dev"), dataset.get_qrels("dev"))
train_end_time = time.time()
print("Train time:", train_end_time-train_start_time)

test_start_time = time.time()
allresultsLM = pt.pipelines.Experiment([BaseLTR_LM],
                                dataset.get_topics("test"),
                                dataset.get_qrels("test"), ["recip_rank", "ndcg_cut_10","map"],
                                names=["LambdaMART"])
test_end_time = time.time()
print("Test time:", test_end_time-test_start_time)
print(allresultsLM)


# Reference:
# [1] Craig Macdonald and Nicola Tonellotto. 2020. Declarative Experimentation inInformation Retrieval using PyTerrier. InProceedings of ICTIR 2020.
