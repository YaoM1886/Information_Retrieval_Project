import pandas as pd
import pyterrier as pt

pt.init()
index = pt.IndexFactory.of("./document_index_without_blocks_and_fields/data.properties")
dfs=[]
dataset = pt.get_dataset("trec-deep-learning-docs")

for mu in [1200, 1500, 1600, 1800, 2000]:
    br = pt.BatchRetrieve(index, wmodel="DirichletLM",properties={"termpipelines": ""},
                          verbose=True, controls={"c":str(mu)})
    df = pt.Experiment([br], dataset.get_topics("dev")[:500], dataset.get_qrels("dev"), eval_metrics=["recip_rank"], names=["Dirichlet mu=%d" % mu])
    dfs.append(df)
print(pd.concat(dfs))
