def prior_LM():
  pt.init()
  dataset = pt.get_dataset("trec-deep-learning-docs")
  index = pt.IndexFactory.of("./document_index_without_blocks_and_fields/data.properties")
  test_query_doc_score_url = pd.read_csv("/Users/sylvia/Downloads/test_corpus/test200-queries-doc-score-url-new.csv",
                                           index_col=0).sort_values(by="docid", ascending=True)
  # define log prior prob
  test_query_doc_score_url["url_prior"] = 0
  test_query_doc_score_url.loc[test_query_doc_score_url["url_depth"] == 3, "url_prior"] = np.log(0.8)
  test_query_doc_score_url.loc[test_query_doc_score_url["url_depth"] == 4, "url_prior"] = np.log(0.15)
  test_query_doc_score_url.loc[test_query_doc_score_url["url_depth"] > 4, "url_prior"] = np.log(0.05)
  
  # this runs an experiment to retrieve results on top100 docs
  DirLM = pt.BatchRetrieve(index, wmodel="DirichletLM", properties={"termpipelines": ""},
                             verbose=True) % 100
  # for each row, add prior to the original LM score
  with_priors_DirLM = DirLM >> pt.apply.doc_score(
           lambda row: test_query_doc_score_url.loc[test_query_doc_score_url["docid"]==row.docid]["url_prior"].values[0]+row.score)
  
  # evaluation process
  query_result = pt.Experiment(
        [with_priors_DirLM],
        dataset.get_topics(type),
        dataset.get_qrels(type),
        eval_metrics=["recip_rank", "map", "ndcg_cut_100"],
        perquery=False)
    
  return query_result
