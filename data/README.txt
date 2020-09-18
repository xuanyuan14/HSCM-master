1. You should generate your own word embedding file (vectors.txt), query candidate dictionary (candidate_qids.json), query word dictionary (qid2wids.json), document word dictionary (did2wids.json), and then put them under the ./data directory.

2. The format of sample session files is as follows:
|-- sessions are splited with '\n\n'
|-- each line within a session is a json-format list: [<qid>, [<did1, did2, ..., did10>], [<clicked1>, <clicked2>, ..., <clicked10>]]
|-- qid: current query identifier
|-- did: document identifier, we only consider top 10 documents for each query
|-- clicked: 0 or 1, whether a document is clicked

3. A sample training script is as follows:
python3 run.py --train --train_dirs ../aol_data/train_sessions.txt --dev_dirs ../aol_data/dev_sessions.txt --test_dirs ../aol_data/test_sessions.txt --num_train_files 1 --num_dev_files 1 --optim adam --eval_freq 1000 --check_point 1000 --learning_rate 1e-3 --weight_decay 1e-6 --dropout_rate 0.2 --batch_size 64 --num_steps 50000 --embed_size 256 --hidden_size 256 --max_d_num 10 --topic_num 2 --topic_len 6 --head_num 4 --cross_qs 3 --bfs_depth 2 --data_type aol_data --model_dir ../aol_data/models_0808_4 --result_dir ../aol_data/results_0808_4 --summary_dir ../aol_data/summary_0808_4 --patience 5