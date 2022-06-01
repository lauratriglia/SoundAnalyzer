import os
import numpy as np


def predict_speaker(embeddings, ls):
    '''
    :param embeddings: .npy file, where the embeddings extracted are stored
    :param ls: to use EmbeddingsHandler class and correctly use get_speaker_db_scan
    :return: speaker_name that is the predicted name of the speaker

    This function takes the embeddings previously created with emb_creation1 and uses a KNN model to return the
    predicted speaker. The KNN model is in speaker_embeddings.py.
    '''
    # ls = EmbeddingsHandler(os.path.join(os.getcwd(), 'DB_s'), n_neighbors=4)
    speaker_name = ls.get_speaker_db_scan(embeddings)

    # if score == -1:
    #     speaker_name = "unknown"

    ls.excluded_entities = []
    print("Predicted speaker name is {}".format(speaker_name))

    return speaker_name
def main():

    main_dir = 'DB_Final'
    all_file_names = os.listdir(main_dir)
    for subdir in all_file_names:
        if os.path.isdir(main_dir + '/' + subdir):
            dir_names = os.listdir(main_dir + '/' + subdir)
            # print(subdir)
        for subdir1 in dir_names:
            if os.path.isdir(main_dir+'/'+subdir+'/'+subdir1):
                files = os.listdir(main_dir+'/'+subdir+'/'+subdir1)
                embs = np.load(main_dir+'/'+subdir+'/'+subdir1+'/'+'chunk.npy').squeeze()
                print(f"EMB outside: {embs}")
                print(subdir1)
                # predict_speaker(np.load(main_dir+'/'+subdir+'/'+subdir1+'/'+'chunk.npy'))
                # print(embs)
                for emb in embs:
                    predict_speaker(emb)

if __name__ == '__main__':
    main()
