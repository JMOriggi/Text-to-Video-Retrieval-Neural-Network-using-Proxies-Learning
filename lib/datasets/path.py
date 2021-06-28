"""
Path to dataset files
"""
class Path(object):
    ## Dataset files
    @staticmethod
    def annotations_file():
        ## [video name] [start time] [end time]##[sentence]
        return ('/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/charades_sta_train.txt',
                '/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/charades_sta_test.txt')
    @staticmethod
    def infos_file():
        ## id,subject,scene,quality,relevance,verified,script,objects,descriptions,actions,length
        return ('/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/charades_sta_infos_train.csv',
                '/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/charades_sta_infos_test.csv')
    @staticmethod
    def video_folder():
        return '/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/charades_480/'
    @staticmethod
    def c3d_model_file():
        return '/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/c3d.pickle'
    @staticmethod
    def embedding_file():
        return '/projectnb/cs591-mm-ml/KuJu/word_embedding/glove.6B.300d.txt'
    
    ## CONFIG init
    @staticmethod
    def config_init_1():
        return '2D-TAN-16x16-K5L8-conv.yaml'
    @staticmethod
    def config_init_2():
        return '2D-TAN-16x16-K5L8-pool.yaml'
    
    
    ## Generated
    @staticmethod
    def word2id_file():
        return '/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/word2id.json'
    @staticmethod
    def video_features_folder():
        return '/projectnb/cs591-mm-ml/KuJu/dataset_charades_sta/video_features/'