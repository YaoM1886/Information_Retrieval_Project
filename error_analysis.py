import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyterrier as pt


def plot_rr_counts(LM_result_path):
    LM_dev_query_result = pd.read_csv(LM_result_path, index_col=0).sort_values(by="value", ascending=False)
    fig, ax = plt.subplots()
    plt.hist(LM_dev_query_result["value"], align="mid", bins=80)
    plt.xlim([0,1])
    plt.xticks(np.arange(0,1.1,0.1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel("reciprocal rank")
    plt.ylabel("counts of queries in each bin")
    plt.show()


def add_url_doclen(query_doc_path):
    pt.init()
    index = pt.IndexFactory.of("./document_index_without_blocks_and_fields/data.properties")
    meta = index.getMetaIndex()
    dev_query_doc = pd.read_csv(query_doc_path, index_col=0)
    dev_query_doc["log_doc_len"] = dev_query_doc["docid"].map(lambda x: np.log(index.getDocumentIndex().getDocumentLength(x)))
    dev_query_doc["url_depth"] = dev_query_doc["docid"].map(lambda x: len(meta.getItem("url", x).split("/"))-1)
    print("Head of dev_query_doc:", dev_query_doc.head())
    print(dev_query_doc["url_depth"].value_counts())
    # dev_query_doc.to_csv("/Users/sylvia/Downloads/dev_query_doc.csv")
    return dev_query_doc

def rank0_and_qrel_doc(LM_result_path, dev_query_doc, qrel_path):
    # for all queries with rank0 documents, length of df should be 5193
    query_doc_rank0 = dev_query_doc[dev_query_doc["rank"]==0]
    query_doc_rank0["query_len"] = query_doc_rank0["query"].map(lambda x: len(x))
    query_rr = pd.read_csv(LM_result_path, index_col=0, usecols=["value", "qid"])
    query_doc_rank0_rr = pd.merge(query_doc_rank0, query_rr, on="qid", how="inner") # get rr for each query
    query_doc_rank0_rr["rr_bin"] = "reciprocal rank >= 0.1"
    query_doc_rank0_rr.loc[query_doc_rank0_rr["value"]<0.1, "rr_bin"] = "reciprocal rank < 0.1"
    print("Length of query_doc_rank0_rr:", len(query_doc_rank0_rr))

    # for all queries with qrel documents
    dev_qrel = pd.read_csv(qrel_path, sep="\t", header=None)
    dev_query_doc_qrel = pd.merge(dev_query_doc, dev_qrel, on=["qid", "docno"], how="inner")
    dev_query_doc_qrel["query_len"] = dev_query_doc_qrel["query"].map(lambda x: len(x))
    print("Length of dev_query_doc_qrel:", len(dev_query_doc_qrel))
    dev_query_doc_qrel_rr = pd.merge(dev_query_doc_qrel, query_rr, on="qid", how="inner")
    dev_query_doc_qrel_rr["rr_bin"] = "reciprocal rank >= 0.1"
    dev_query_doc_qrel_rr.loc[dev_query_doc_qrel_rr["value"]<0.1, "rr_bin"] = "reciprocal rank < 0.1"
    print("Length of dev_query_doc_qrel_rr:", len(dev_query_doc_qrel_rr))

    return query_doc_rank0_rr, dev_query_doc_qrel_rr, query_rr


def extract_avg_doc(dev_query_doc, query_rr, agg_type):
    # for all queries with average rank100 documents
    avg_query_doc = dev_query_doc.groupby("qid").agg({agg_type:"mean"})
    avg_query_doc["query_len"] = avg_query_doc["query"].map(lambda x: len(x))
    query_doc_avg_rr = pd.merge(avg_query_doc, query_rr, on="qid", how="inner") # get rr for each query
    query_doc_avg_rr["rr_bin"] = "reciprocal rank >= 0.1"
    query_doc_avg_rr.loc[query_doc_avg_rr["value"]<0.1, "rr_bin"] = "reciprocal rank < 0.1"
    print("Length of query_doc_avg_rr:", len(query_doc_avg_rr))
    return query_doc_avg_rr


def groupby_rr_bin(query_doc_rank0_rr, dev_query_doc_qrel_rr, query_doc_avg_rr, group_type, ylabel):
    fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    groups = query_doc_rank0_rr.groupby("rr_bin")
    plt.subplot(1,3,1)
    for name, group in groups:
        plt.plot(group["query_len"], group[group_type], marker="o", linestyle="", label=name)

    plt.subplot(1,3,2)
    groups = dev_query_doc_qrel_rr.groupby("rr_bin")
    for name, group in groups:
        plt.plot(group["query_len"], group[group_type], marker="o", linestyle="", label=name)

    plt.subplot(1,3,3)
    groups = query_doc_avg_rr.groupby("rr_bin")
    for name, group in groups:
        plt.plot(group["query_len"], group[group_type], marker="o", linestyle="", label=name)

    plt.xlabel("query length")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

