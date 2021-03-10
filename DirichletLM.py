import time
import pandas as pd
import numpy as np
import pyterrier as pt
import ir_datasets

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


DirichletLM_br = pt.BatchRetrieve(index, wmodel="DirichletLM", properties={"termpipelines": ""}, verbose=True) % 100

# # with Sequential Dependence
# sdm = pt.rewrite.SequentialDependence()
# dph = pt.BatchRetrieve(index, wmodel="DirichletLM", properties={"termpipelines": ""}, verbose=True) % 100
# DirichletLM_br = sdm >> dph
# #


start_evaluation_time = time.time()
pt.Experiment([DirichletLM_br], dataset.get_topics("test"), dataset.get_qrels("test"), eval_metrics=["recip_rank", "ndcg_cut_10","map"])
print("evaluation time: ")
print(time.time() - start_evaluation_time)



# Reference:
# [1] Craig Macdonald and Nicola Tonellotto. 2020. Declarative Experimentation inInformation Retrieval using PyTerrier. InProceedings of ICTIR 2020.
