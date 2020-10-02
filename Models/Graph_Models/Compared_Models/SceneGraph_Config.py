import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU or not')
    parser.add_argument("--evaluate_complex_data", action="store_true", help='evaluate on complex ot whole data')

    parser.add_argument('--QG_model_file', type=str, default='../../Model_Data/SceneGraph_QG/graph_model_base', help='directory of question generation model')
    # parser.add_argument('--QG_model_file', type=str, default='../../Model_Data/SceneGraph_QG/graph_model_up', help='directory of question generation model')
    # parser.add_argument('--QG_model_file', type=str, default='../../Model_Data/SceneGraph_QG/graph_model_pipe', help='directory of question generation model')
    # parser.add_argument('--QG_model_file', type=str, default='../../Model_Data/SceneGraph_QG/as_model', help='directory of question generation model')
    # parser.add_argument('--QG_model_file', type=str, default='../../Model_Data/SceneGraph_QG/af_model', help='directory of question generation model')

    parser.add_argument('--src_vocab_size', type=int, default=35865, help='size of src vocabulary')
    parser.add_argument('--tgt_vocab_size', type=int, default=13832, help='size of tgt vocabulary')
    parser.add_argument('--embedding_size', type=int, default=300, help='size of word embedding')
    parser.add_argument('--type_embedding_dense_size', type=int, default=20, help='size of word embedding')
    parser.add_argument('--tagging_embedding_size', type=int, default=20, help='size of tagging embedding')

    parser.add_argument('--encoder_hidden_size', type=int, default=300, help='size of encoder hidden states')
    parser.add_argument('--tgt_encoder_hidden_size', type=int, default=300, help='size of tgt encoder hidden states')
    parser.add_argument('--node_hidden_size', type=int, default=150, help='size of node hidden states')
    parser.add_argument('--path_hidden_size', type=int, default=300, help='size of path hidden states')
    parser.add_argument('--entity_encoder_hidden_size', type=int, default=300, help='size of entity encoder hidden states')
    parser.add_argument('--selection_entity_encoder_hidden_size', type=int, default=300, help='size of entity encoder hidden states')
    parser.add_argument('--decoder_hidden_size', type=int, default=600, help='size of decoder hidden states')
    parser.add_argument('--attn_size', type=int, default=300, help='size of attention states')
    parser.add_argument('--entity_dense_size', type=int, default=600, help='size of entity dense units')
    parser.add_argument('--question_dense_size', type=int, default=300, help='size of dense units')
    parser.add_argument('--answer_class_num', type=int, default=3, help='number of answer tag')
    parser.add_argument('--end_tag_num', type=int, default=2, help='number of answer end tag')
    parser.add_argument('--graph_tag_num', type=int, default=3, help='number of graph tag')
    parser.add_argument('--entity_tag_num', type=int, default=2, help='number of entity tag: exist or not')
    parser.add_argument('--distance_tag_num', type=int, default=6, help='number of distance tag')
    parser.add_argument('--num_layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--keep_neighbor_distance', type=int, default=2, help='only attention to neighbors in distance')

    parser.add_argument('--attn_input_feed', type=bool, default=False, help='feed attn context into input')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epoches', type=int, default=20, help='number of epoches')
    parser.add_argument('--selection_num_epoches', type=int, default=20, help='number of epoches')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate of training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.96, help='decay learning rate')
    parser.add_argument('--clip_grad', type=int, default=5, help='clip gradients')
    parser.add_argument('--beam_size', type=int, default=5, help='size of beam search')
    parser.add_argument('--max_len', type=int, default=30, help='length of generated questions')
    parser.add_argument('--length_penalty_factor', type=float, default=1.0, help='length of generated questions')
    parser.add_argument('--pad_id', type=int, default=0, help='pad id')
    parser.add_argument('--eos_id', type=int, default=2, help='eos id')

    parser.add_argument('--l1_reg', type=float, default=1e-6, help='l1 regularzation')
    parser.add_argument('--l2_reg', type=float, default=1e-6, help='l2_regularzation')

    parser.add_argument('--src_embedding_file', type=str, default='../../Data/processed/SQuAD1.0/Graph_Analysis/SceneGraph/embeddings_src.npy')
    parser.add_argument('--tgt_embedding_file', type=str, default='../../Data/processed/SQuAD1.0/Graph_Analysis/SceneGraph/embeddings_tgt.npy')

    return parser.parse_args()