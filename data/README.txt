1. You should generate your own word embedding file (vectors.txt), query candidate dictionary (candidate_qids.json), query word dictionary (qid2wids.json), document word dictionary (did2wids.json), and then put them under the ./data directory.

2. The format of sample session files is as follows:
|-- sessions are splited with '\n\n'
|-- each line within a session is a json-format list: [<qid>, [<did1, did2, ..., did10>], [<clicked1>, <clicked2>, ..., <clicked10>]]
|-- qid: current query identifier
|-- did: document identifier, we only consider top 10 documents for each query
|-- clicked: 0 or 1, whether a document is clicked

3. A sample training script is as follows:
python3 -u run.py --train --train_dirs ../data/sample_train_sess.txt \--dev_dirs ../data/sample_dev_sess.txt --test_dirs ../data/sample_test_sess.txt --num_train_files 1 --num_dev_files 1 --num_test_files 1 --optim adam --eval_freq 5 --check_point 5 --learning_rate 1e-3 --weight_decay 1e-4 --dropout_rate 0.2 --batch_size 8 --num_steps 100 --embed_size 256 --hidden_size 256 --max_d_num 10 --topic_len 24 --model_dir ../data/models_1031 --result_dir ../data/results_1031 --summary_dir ../data/summary_1031 --patience 5